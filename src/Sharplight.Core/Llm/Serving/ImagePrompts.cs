using System.Runtime.InteropServices;
using System.Security.Cryptography;
using Sharplight.Tokenizers;
using Sharplight.Vision;

namespace Sharplight.Llm.Serving;

/// <summary>
/// Image prompts for the <see cref="InferenceEngine"/>: expands the image placeholders of a tokenized prompt into the rows the architecture
/// defines (<see cref="IMultimodalArchitecture"/>), with each image as <see cref="GenerationRequest.PromptMedia"/> whose rows the engine encodes
/// with this object's <see cref="VisionEncoder"/> right before the prompt chunk that needs them, and the M-RoPE positions as
/// <see cref="GenerationRequest.PromptPositions"/>. The engine's decoder must take media rows (<see cref="LlmModel.CreateBatchDecoder"/> with
/// media input). <see cref="WithImages"/> is thread-safe (preprocessing and hashing run on the caller's thread); the engine alone calls
/// <see cref="Encode"/>.
/// </summary>
public sealed class ImagePrompts : IMediaEncoder, IDisposable
{
    private readonly VisionEncoder _encoder;
    private readonly ITokenizer _tokenizer;
    private readonly IMultimodalArchitecture _architecture;
    private readonly bool _ownsEncoder;

    /// <summary>Creates the prompt builder over a model's vision encoder.</summary>
    /// <param name="encoder">The vision encoder (<see cref="LlmModel.CreateVisionEncoder"/>).</param>
    /// <param name="tokenizer">The model's tokenizer (finds the placeholder tokens).</param>
    /// <param name="ownsEncoder">Dispose <paramref name="encoder"/> with this object.</param>
    /// <exception cref="NotSupportedException">The architecture has no multimodal prompt assembly.</exception>
    public ImagePrompts(VisionEncoder encoder, ITokenizer tokenizer, bool ownsEncoder = false)
    {
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentNullException.ThrowIfNull(tokenizer);
        _encoder = encoder;
        _tokenizer = tokenizer;
        _ownsEncoder = ownsEncoder;
        _architecture = encoder.Multimodal
            ?? throw new NotSupportedException($"Architecture '{encoder.Definition.Architecture}' has no multimodal prompt support.");
    }

    /// <summary>The vision encoder.</summary>
    public VisionEncoder Encoder => _encoder;

    /// <inheritdoc/>
    public int EmbeddingWidth => _encoder.EmbeddingSize;

    /// <summary>Device memory of the vision encoder (its weights and activations).</summary>
    public long DeviceMemoryBytes => _encoder.Compiled.AcceleratorMemoryBytes;

    /// <summary>
    /// The request with its prompt's image placeholders expanded: <see cref="GenerationRequest.PromptTokens"/> holds one placeholder token per
    /// image (in order, as the chat template renders them); the result has one row per image embedding, the images as
    /// <see cref="GenerationRequest.PromptMedia"/>, and the architecture's positions and media weights. A request without images is returned as is.
    /// </summary>
    /// <param name="request">The request with placeholder tokens.</param>
    /// <param name="images">The images, in placeholder order.</param>
    /// <exception cref="ArgumentException">The number of placeholders differs from the number of images.</exception>
    public GenerationRequest WithImages(GenerationRequest request, IReadOnlyList<ImageInput> images)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(images);
        if (images.Count == 0)
        {
            return request;
        }

        var preprocessed = images.Select(_encoder.Preprocess).ToList();
        var prompt = _architecture.Assemble(new MultimodalContext(_encoder.Definition, LookupToken), request.PromptTokens, preprocessed, 0);
        var media = new List<PromptMedia>(images.Count);
        int row = 0;
        while (row < prompt.Length)
        {
            int image = prompt.ImageIndex[row];
            if (image < 0)
            {
                row++;
                continue;
            }

            int start = row;
            while (row < prompt.Length && prompt.ImageIndex[row] == image)
            {
                if (prompt.ImageRow[row] != row - start)
                {
                    throw new NotSupportedException("The architecture lays an image's rows out of order; the engine takes them in encoder order.");
                }

                row++;
            }

            var content = new ImageMedia(this, preprocessed[image], Hash(preprocessed[image]));
            if (content.Rows != row - start)
            {
                throw new NotSupportedException($"The architecture placed {row - start} rows of an image whose encoder gives {content.Rows}.");
            }

            media.Add(new PromptMedia(start, content));
        }

        return request with
        {
            PromptTokens = prompt.Tokens,
            PromptMedia = media,
            PromptPositions = prompt.Positions,
            NextPosition = prompt.NextPosition,
            PromptMediaWeights = prompt.MediaWeights,
        };
    }

    /// <inheritdoc/>
    /// <exception cref="ArgumentException">A content was not created by this object.</exception>
    public IReadOnlyList<float[]> Encode(IReadOnlyList<MediaContent> contents)
    {
        ArgumentNullException.ThrowIfNull(contents);
        var rows = new float[contents.Count][];
        for (int i = 0; i < contents.Count; i++)
        {
            if (contents[i] is not ImageMedia image || image.Owner != this)
            {
                throw new ArgumentException("The content was not created by this image prompt builder.", nameof(contents));
            }

            rows[i] = _encoder.Encode(image.Image);
        }

        return rows;
    }

    /// <summary>Disposes the vision encoder when owned.</summary>
    public void Dispose()
    {
        if (_ownsEncoder)
        {
            _encoder.Dispose();
        }
    }

    private int? LookupToken(string text)
    {
        var ids = _tokenizer.Encode(text, addSpecial: false, parseSpecial: true);
        return ids.Length == 1 ? ids[0] : null;
    }

    /// <summary>128 bits of SHA-256 of everything the encoder's output depends on: the preprocessed pixels, the grid and the output layout.</summary>
    private static UInt128 Hash(PreprocessedImage image)
    {
        using var sha = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        Span<int> header = [image.Width, image.Height, image.GridHeight, image.GridWidth, image.MergedHeight, image.MergedWidth, image.TileColumns, image.TileRows];
        sha.AppendData(MemoryMarshal.AsBytes(header));
        sha.AppendData(MemoryMarshal.AsBytes(image.Patches.AsSpan()));
        sha.AppendData(MemoryMarshal.AsBytes(image.Layout.AsSpan()));
        Span<byte> digest = stackalloc byte[32];
        sha.GetHashAndReset(digest);
        return MemoryMarshal.Read<UInt128>(digest);
    }
}
