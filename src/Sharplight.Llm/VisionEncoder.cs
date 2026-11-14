using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Vision;

namespace Sharplight.Llm;

/// <summary>
/// A compiled <see cref="WellKnown.VisionEncodeGraph"/> plus the generic image preprocessing driven by the model's
/// canonical <see cref="VisionParameters"/>: raw RGB pixels in, one embedding row per merged image patch out.
/// Create with <see cref="LlmModel.CreateVisionEncoder"/>.
/// </summary>
public sealed class VisionEncoder : IDisposable
{
    private readonly CompiledModel _compiled;
    private readonly CompiledModel? _tile;
    private readonly RunParameters _run = new();
    private readonly Value? _patches, _positions, _embIndex, _embWeight;
    private readonly Value? _windowSpans, _windowOrder;
    private readonly string _output = VisionParameters.EmbeddingsOutput;

    /// <summary>The model the encoder belongs to.</summary>
    public ModelDefinition Definition { get; }
    /// <summary>Canonical vision-tower parameters (patch size, merge unit, image statistics).</summary>
    public VisionParameters Parameters { get; }
    /// <summary>The image preprocessing used by <see cref="Preprocess"/>.</summary>
    public VisionPreprocessor Preprocessor { get; }
    /// <summary>Largest number of patches one image may have (the compiled bound).</summary>
    public int MaxPatches { get; }
    /// <summary>The compiled vision graph.</summary>
    public CompiledModel Compiled => _compiled;

    /// <summary>
    /// Compiles the vision encoder of a built model onto devices. <paramref name="limits"/> bound the image size: the encoder
    /// handles images of up to <c>MaxTokens</c> (default 1024) output tokens, larger images are resized to fit.
    /// </summary>
    public static VisionEncoder Create(ModelDefinition def, IReadOnlyList<DeviceSlot> devices, DeviceSlot? fallback = null, ImageTokenLimits? limits = null,
        IReadOnlySet<string>? disabledFeatures = null, Func<IComputeDevice, ICompilationCache?>? cacheFor = null)
    {
        var p = VisionParameters.Of(def) ?? throw new NotSupportedException("This model has no vision tower (for GGUF models pass the projector file as the \"mmproj\" auxiliary file).");
        if (p.Sam is not null)
        {
            // Fixed-size views: static graphs, one run per view.
            CompiledModel CompileView(string graph) => CompiledModel.Compile(def.Graph(graph), new PlacementOptions
            {
                Devices = devices,
                Fallback = fallback,
                Compile = new CompileOptions { MaxTokens = 1, ContextLength = 1, DisabledFeatures = disabledFeatures ?? new HashSet<string>(), CacheFor = cacheFor },
            });
            var global = CompileView(WellKnown.VisionEncodeGraph);
            var tile = def.Graphs.ContainsKey(SamParameters.TileGraph) ? CompileView(SamParameters.TileGraph) : null;
            return new VisionEncoder(def, p, global, tile, new VisionPreprocessor(p), 0);
        }

        int maxTokens = limits?.MaxTokens ?? 1024;
        int maxPatches = maxTokens * p.MergeUnit;
        var compiled = CompiledModel.Compile(def.Graph(WellKnown.VisionEncodeGraph), new PlacementOptions
        {
            Devices = devices,
            Fallback = fallback,
            Compile = new CompileOptions
            {
                MaxTokens = maxPatches,
                ContextLength = 1,
                SymbolBounds = new Dictionary<string, long> { [VisionParameters.PatchSymbol] = maxPatches, [VisionParameters.MergedSymbol] = maxTokens },
                DisabledFeatures = disabledFeatures ?? new HashSet<string>(),
                CacheFor = cacheFor,
            },
        });
        var pre = new VisionPreprocessor(p, new ImageTokenLimits(limits?.MinTokens, maxTokens));
        return new VisionEncoder(def, p, compiled, null, pre, maxPatches);
    }

    private VisionEncoder(ModelDefinition def, VisionParameters p, CompiledModel compiled, CompiledModel? tile, VisionPreprocessor pre, int maxPatches)
    {
        Definition = def;
        Parameters = p;
        _compiled = compiled;
        _tile = tile;
        Preprocessor = pre;
        MaxPatches = maxPatches;
        var g = compiled.Graph;
        _patches = g.Inputs.FirstOrDefault(v => v.Input!.Name == VisionParameters.PatchesInput);
        _positions = g.Inputs.FirstOrDefault(v => v.Input!.Name == VisionParameters.PositionsInput);
        _embIndex = g.Inputs.FirstOrDefault(v => v.Input!.Name == VisionParameters.EmbeddingIndexInput)!;
        _embWeight = g.Inputs.FirstOrDefault(v => v.Input!.Name == VisionParameters.EmbeddingWeightInput)!;
        _windowSpans = g.Inputs.FirstOrDefault(v => v.Input!.Name == VisionParameters.WindowSpansInput);
        _windowOrder = g.Inputs.FirstOrDefault(v => v.Input!.Name == VisionParameters.WindowOrderInput);
    }

    /// <summary>Width of one embedding row: the language model's hidden size, times 1 + the number of DeepStack layers.</summary>
    public int EmbeddingSize => Parameters.EmbeddingWidth;

    /// <summary>Resizes, normalizes and patchifies an image (see <see cref="VisionPreprocessor"/>).</summary>
    public PreprocessedImage Preprocess(ImageInput image) => Preprocessor.Process(image);

    /// <summary>Preprocesses and encodes an image: [tokens, <see cref="EmbeddingSize"/>] row-major.</summary>
    public float[] Encode(ImageInput image) => Encode(Preprocess(image));

    /// <summary>Encodes a preprocessed image: [tokens, <see cref="EmbeddingSize"/>] row-major.</summary>
    /// <exception cref="InvalidOperationException">The image has more patches than the encoder was compiled for.</exception>
    public float[] Encode(PreprocessedImage image)
    {
        if (Parameters.Sam is not null)
        {
            return EncodeViews(image);
        }

        int n = image.PatchCount;
        int merged = n / Parameters.MergeUnit;
        if (n > MaxPatches)
        {
            throw new InvalidOperationException($"The image has {n} patches but the encoder was compiled for at most {MaxPatches}; lower the image resolution or raise the image token limit.");
        }

        _compiled.SetInput<float>(_patches!, image.Patches);
        _compiled.SetInput<int>(_positions!, image.Positions);
        _compiled.SetInput<int>(_embIndex!, image.EmbeddingIndex);
        _compiled.SetInput<float>(_embWeight!, image.EmbeddingWeight);
        if (_windowSpans is not null && _windowOrder is not null)
        {
            if (image.WindowSpans.Length != 2 * n || image.WindowOrder.Length != merged)
            {
                throw new ArgumentException("The image was not preprocessed for this encoder's windowed attention (window spans and order missing).", nameof(image));
            }

            _compiled.SetInput<int>(_windowSpans, image.WindowSpans);
            _compiled.SetInput<int>(_windowOrder, image.WindowOrder);
        }

        _run.Symbols[VisionParameters.PatchSymbol] = n;
        _run.Symbols[VisionParameters.MergedSymbol] = merged;
        _compiled.Run(_run);
        var result = new float[(long)merged * EmbeddingSize];
        _compiled.GetOutput<float>(_output, result);
        return result;
    }

    /// <summary>
    /// Two-stage towers: encodes the global view and each tile, then lays the rows out (see <see cref="SamParameters"/>): the tiles stitched
    /// into one grid with a newline after each row, the global grid with a newline after each row, the view separator.
    /// </summary>
    private float[] EncodeViews(PreprocessedImage image)
    {
        var sam = Parameters.Sam!;
        int width = EmbeddingSize;
        int globalLen = 3 * sam.GlobalSize * sam.GlobalSize;
        int tileLen = 3 * sam.TileSize * sam.TileSize;
        int tiles = image.TileColumns * image.TileRows;
        if (image.Patches.Length != globalLen + tiles * tileLen || image.TokenCount != image.MergedWidth || image.MergedHeight != 1)
        {
            throw new ArgumentException("The image was not preprocessed for this encoder's views (global view and tiles).", nameof(image));
        }

        var result = new float[(long)image.TokenCount * width];
        int row = 0;
        var newline = new float[width];
        var separator = new float[width];
        if (tiles > 0)
        {
            var model = _tile ?? _compiled;
            int ft = sam.FeatureSide(sam.TileSize, Parameters.PatchSize);
            var views = new float[tiles][];
            for (int t = 0; t < tiles; t++)
            {
                views[t] = RunView(model, image.Patches.AsSpan(globalLen + t * tileLen, tileLen), ft * ft, newline, separator);
            }

            for (int tr = 0; tr < image.TileRows; tr++)
            {
                for (int y = 0; y < ft; y++)
                {
                    for (int tc = 0; tc < image.TileColumns; tc++)
                    {
                        views[tr * image.TileColumns + tc].AsSpan(y * ft * width, ft * width).CopyTo(result.AsSpan(row * width));
                        row += ft;
                    }

                    newline.CopyTo(result.AsSpan(row++ * width));
                }
            }
        }

        int f = sam.FeatureSide(sam.GlobalSize, Parameters.PatchSize);
        var globalRows = RunView(_compiled, image.Patches.AsSpan(0, globalLen), f * f, newline, separator);
        for (int y = 0; y < f; y++)
        {
            globalRows.AsSpan(y * f * width, f * width).CopyTo(result.AsSpan(row * width));
            row += f;
            newline.CopyTo(result.AsSpan(row++ * width));
        }

        separator.CopyTo(result.AsSpan(row++ * width));
        return result;
    }

    private float[] RunView(CompiledModel model, ReadOnlySpan<float> pixels, int rows, float[] newline, float[] separator)
    {
        var input = model.Graph.Inputs.First(v => v.Input!.Name == SamParameters.PixelsInput);
        model.SetInput<float>(input, pixels);
        model.Run(_run);
        var embeddings = new float[(long)rows * EmbeddingSize];
        model.GetOutput<float>(VisionParameters.EmbeddingsOutput, embeddings);
        model.GetOutput<float>(SamParameters.NewlineOutput, newline);
        model.GetOutput<float>(SamParameters.SeparatorOutput, separator);
        return embeddings;
    }

    /// <summary>Releases the device memory of the encoder.</summary>
    public void Dispose()
    {
        _compiled.Dispose();
        _tile?.Dispose();
    }
}
