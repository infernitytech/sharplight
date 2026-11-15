using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.WhisperCpp;

/// <summary>
/// whisper.cpp importer: maps whisper.cpp model files (<c>ggml-*.bin</c>: f32, f16 and the quantized variants such as q5_1, q8_0) onto
/// the canonical <see cref="ModelSource"/> of a Whisper-family encoder-decoder (canonical architecture <c>encoder_decoder</c> with an audio
/// encoder, the same canonical model a Hugging Face Whisper checkpoint imports to). Weight data is shared zero-copy from the memory-mapped
/// file; only names change. The stored mel filter bank becomes the <c>feature-extractor/mel-filters</c> attachment, and the vocabulary is
/// rebuilt into a canonical byte-level BPE tokenizer with Whisper's special tokens and decoding keys (see <see cref="WhisperCppVocabulary"/>).
/// </summary>
/// <remarks>Register once with <see cref="Register"/> so that <see cref="ModelImporterRegistry"/> opens whisper.cpp files, or call <see cref="Open"/>.</remarks>
public sealed partial class WhisperCppImporter : IModelImporter
{
    private const string Architecture = "encoder_decoder";
    private const int SampleRate = 16000;
    private const int HopLength = 160;
    private const int ChunkSeconds = 30;

    /// <inheritdoc/>
    public string Format => "whisper.cpp";

    /// <summary>Registers the importer with <see cref="ModelImporterRegistry"/>.</summary>
    public static void Register() => ModelImporterRegistry.Register(new WhisperCppImporter());

    /// <inheritdoc/>
    /// <remarks>True when <paramref name="path"/> is an existing file starting with the ggml magic of whisper.cpp model files.</remarks>
    public bool CanOpen(string path) => WhisperCppFile.HasMagic(path);

    /// <inheritdoc/>
    ModelSource IModelImporter.Open(string path) => Open(path);

    /// <summary>Opens a whisper.cpp model file.</summary>
    /// <param name="path">The <c>.bin</c> file.</param>
    /// <exception cref="InvalidDataException">The file is not a well-formed whisper.cpp model.</exception>
    /// <exception cref="NotSupportedException">The model uses a tensor type or layout that is not supported.</exception>
    public static ModelSource Open(string path)
    {
        var file = WhisperCppFile.Open(path);
        try
        {
            var hp = file.HyperParameters;
            if (hp.AudioWidth != hp.TextWidth || hp.AudioHeads != hp.TextHeads)
            {
                throw new NotSupportedException($"{path}: encoder and decoder of different widths or head counts are not supported.");
            }

            if (file.MelBins != hp.MelBins)
            {
                throw new InvalidDataException($"{path}: the filter bank has {file.MelBins} mel bins, the model {hp.MelBins}.");
            }

            var metadata = HyperParameters(hp, (file.FrequencyBins - 1) * 2);
            metadata[CanonicalKeys.Architecture] = Architecture;
            metadata[CanonicalKeys.Name] = Path.GetFileNameWithoutExtension(path);
            WhisperCppVocabulary.AddTo(file, metadata);
            return new ModelSource
            {
                Format = "whisper.cpp",
                Architecture = Architecture,
                Metadata = metadata,
                Weights = Weights(file),
                Attachments = new Dictionary<string, IWeightSource>(StringComparer.Ordinal)
                {
                    ["feature-extractor/mel-filters"] = new MemoryWeightSource(file.MelFilters),
                },
                OwnedResources = [file],
                Path = path,
            };
        }
        catch
        {
            file.Dispose();
            throw;
        }
    }

    private static Dictionary<string, object> HyperParameters(WhisperCppHyperParameters hp, int fftSize)
    {
        string a = Architecture + ".";
        int headDim = hp.TextWidth / hp.TextHeads;
        return new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "hidden_size"] = hp.TextWidth,
            [a + "encoder.layer_count"] = hp.AudioLayers,
            [a + "decoder.layer_count"] = hp.TextLayers,
            [a + "attention.head_count"] = hp.TextHeads,
            [a + "attention.head_dim"] = headDim,
            [a + "attention.scale"] = 1f / MathF.Sqrt(headDim),
            [a + "norm"] = "layer",
            [a + "norm_eps"] = 1e-5f,
            [a + "pre_norm"] = true,
            [a + "mlp.activation"] = "gelu",
            [a + "decoder_start_token"] = new WhisperCppSpecialTokens(hp).StartOfTranscript,
            [a + "context_length"] = hp.TextContext,
            [a + "encoder.input"] = "audio",
            [a + "encoder.conv.strides"] = new long[] { 1, 2 },
            [a + "encoder.rows"] = hp.AudioContext,
            [a + "cross_attention.cached"] = true,
            [a + "audio.sample_rate"] = SampleRate,
            [a + "audio.fft_size"] = fftSize,
            [a + "audio.hop_length"] = HopLength,
            [a + "audio.mel_bins"] = hp.MelBins,
            [a + "audio.chunk_samples"] = ChunkSeconds * SampleRate,
        };
    }

    [GeneratedRegex(@"^(encoder|decoder)\.blocks\.(\d+)\.(.+)\.(weight|bias)$")]
    private static partial Regex BlockName();

    /// <summary>Per-block modules, whisper.cpp name to canonical name.</summary>
    private static readonly Dictionary<string, string> s_blockModules = new(StringComparer.Ordinal)
    {
        ["attn_ln"] = "attention_norm",
        ["attn.query"] = "attention.q",
        ["attn.key"] = "attention.k",
        ["attn.value"] = "attention.v",
        ["attn.out"] = "attention.o",
        ["cross_attn_ln"] = "cross_attention_norm",
        ["cross_attn.query"] = "cross_attention.q",
        ["cross_attn.key"] = "cross_attention.k",
        ["cross_attn.value"] = "cross_attention.v",
        ["cross_attn.out"] = "cross_attention.o",
        ["mlp_ln"] = "mlp_norm",
        ["mlp.0"] = "mlp.up",
        ["mlp.2"] = "mlp.down",
    };

    /// <summary>Whole-model tensors, whisper.cpp name to canonical name.</summary>
    private static readonly Dictionary<string, string> s_modelTensors = new(StringComparer.Ordinal)
    {
        ["encoder.positional_embedding"] = "encoder.position_embedding.weight",
        ["decoder.positional_embedding"] = "decoder.position_embedding.weight",
        ["decoder.token_embedding.weight"] = "token_embedding.weight",
        ["encoder.conv1.weight"] = "encoder.conv.0.weight",
        ["encoder.conv1.bias"] = "encoder.conv.0.bias",
        ["encoder.conv2.weight"] = "encoder.conv.1.weight",
        ["encoder.conv2.bias"] = "encoder.conv.1.bias",
        ["encoder.ln_post.weight"] = "encoder.output_norm.weight",
        ["encoder.ln_post.bias"] = "encoder.output_norm.bias",
        ["decoder.ln.weight"] = "decoder.output_norm.weight",
        ["decoder.ln.bias"] = "decoder.output_norm.bias",
    };

    private static Dictionary<string, WeightInfo> Weights(WhisperCppFile file)
    {
        var weights = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
        foreach (var tensor in file.Tensors.Values)
        {
            string canonical = CanonicalName(tensor.Name);
            var shape = tensor.Shape;
            if (canonical.EndsWith(".bias", StringComparison.Ordinal) && shape.Rank == 2)
            {
                // whisper.cpp stores the convolution biases as [channels, 1].
                shape = new Shape(shape.Elements());
            }

            weights[canonical] = new WeightInfo
            {
                Name = canonical,
                Type = tensor.Type,
                Shape = shape,
                Source = new WhisperCppWeightSource(file, tensor.Offset, tensor.Length),
            };
        }

        return weights;
    }

    private static string CanonicalName(string name)
    {
        if (s_modelTensors.TryGetValue(name, out var canonical))
        {
            return canonical;
        }

        var match = BlockName().Match(name);
        if (!match.Success || !s_blockModules.TryGetValue(match.Groups[3].Value, out var module))
        {
            throw new NotSupportedException($"whisper.cpp: tensor '{name}' has no canonical mapping.");
        }

        return $"{match.Groups[1].Value}.layers.{match.Groups[2].Value}.{module}.{match.Groups[4].Value}";
    }
}
