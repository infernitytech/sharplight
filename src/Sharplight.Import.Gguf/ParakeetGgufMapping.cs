using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using Sharplight.Audio;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.Gguf;

/// <summary>
/// GGUF <c>parakeet</c> files as transcribe.cpp writes them (<c>stt.parakeet.*</c> hyper-parameters, <c>stt.frontend.*</c> log-mel settings,
/// the SentencePiece vocabulary with the blank as its last token, <c>enc.*</c>, <c>pred.*</c> and <c>joint.*</c> tensors; the repositories
/// <c>handy-computer/parakeet-*-gguf</c>) → canonical <c>conformer_asr</c>, the model a Hugging Face Parakeet checkpoint imports to. The
/// file stores no filter bank, so the Slaney bank of the front end's parameters becomes the <c>mel-filters</c> attachment. Transducer
/// heads only (TDT, RNN-T); the LSTM's single bias per layer maps to the input bias.
/// </summary>
internal sealed partial class ParakeetGgufMapping : IGgufArchitectureMapping
{
    private const string Parakeet = "stt.parakeet.";
    private const string Frontend = "stt.frontend.";
    private const string MelFilters = "feature-extractor/mel-filters";
    private const int WindowSeconds = 30;

    /// <inheritdoc/>
    public IReadOnlyList<string> GgufArchitectures { get; } = ["parakeet"];

    /// <inheritdoc/>
    public string Architecture => "conformer_asr";

    /// <inheritdoc/>
    public IEnumerable<string> Validate(GgufMetadataView g)
    {
        string head = Text(g, Parakeet + "head_kind", "");
        if (head is not ("tdt" or "rnnt"))
        {
            yield return $"The parakeet head '{head}' is not supported (transducer heads tdt and rnnt only).";
        }

        if (Int(g, Parakeet + "encoder.subsampling_factor", 8) != 8)
        {
            yield return "Only the 8x subsampling of the FastConformer is supported.";
        }

        if (Text(g, Parakeet + "encoder.conv_norm_type", "batch_norm") != "batch_norm")
        {
            yield return $"The convolution norm '{Text(g, Parakeet + "encoder.conv_norm_type", "")}' is not supported (batch norm only).";
        }

        if (Text(g, Parakeet + "encoder.att_context_style", "regular") is not ("regular" or "chunked_limited_with_rc") || Int(g, Parakeet + "encoder.conv_context_right", 4) != 4)
        {
            yield return $"The cache-aware streaming encoder ('{Text(g, Parakeet + "encoder.att_context_style", "")}' attention, causal convolutions) is not supported.";
        }

        if (Text(g, Frontend + "type", "mel") != "mel")
        {
            yield return $"The audio front end '{Text(g, Frontend + "type", "")}' is not supported.";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(GgufMetadataView g)
    {
        string a = Architecture + ".";
        int hidden = Int(g, Parakeet + "encoder.d_model");
        long channels = Int(g, Parakeet + "encoder.subsampling_channels", 256);
        int sampleRate = Int(g, Frontend + "sample_rate", 16000);
        int hop = Int(g, Frontend + "hop_length", 160);
        string head = Text(g, Parakeet + "head_kind", "tdt");
        var result = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "hidden_size"] = hidden,
            [a + "encoder.layer_count"] = Int(g, Parakeet + "encoder.n_layers"),
            [a + "encoder.attention.head_count"] = Int(g, Parakeet + "encoder.n_heads"),
            [a + "encoder.subsampling.strides"] = new long[] { 2, 2, 1, 2, 1 },
            [a + "encoder.subsampling.groups"] = new long[] { 1, channels, 1, channels, 1 },
            [a + "encoder.subsampling.activate"] = new[] { true, false, true, false, true },
            [a + "encoder.input_scale"] = Int(g, Parakeet + "encoder.xscaling", 0) != 0 ? MathF.Sqrt(hidden) : 1f,
            [a + "encoder.mlp.activation"] = "silu",
            [a + "encoder.norm_eps"] = 1e-5f,
            [a + "encoder.conv.norm_eps"] = 1e-5f,
            [a + "audio.features"] = "log-mel",
            [a + "audio.sample_rate"] = sampleRate,
            [a + "audio.fft_size"] = Int(g, Frontend + "n_fft", 512),
            [a + "audio.window_size"] = Int(g, Frontend + "win_length", 400),
            [a + "audio.periodic_window"] = false,
            [a + "audio.hop_length"] = hop,
            [a + "audio.mel_bins"] = Int(g, Frontend + "num_mels", 128),
            [a + "audio.preemphasis"] = Double.Parse(Convert.ToString(g.All.GetValueOrDefault(Frontend + "pre_emphasis", 0.97), System.Globalization.CultureInfo.InvariantCulture)!, System.Globalization.CultureInfo.InvariantCulture),
            [a + "audio.pad_mode"] = "constant",
            [a + "audio.log_base"] = "e",
            [a + "audio.log_guard"] = Math.Pow(2, -24),
            [a + "audio.dynamic_range"] = 0.0,
            [a + "audio.log_offset"] = 0.0,
            [a + "audio.log_scale"] = 1.0,
            [a + "audio.mel_normalize"] = "per-channel",
            [a + "audio.variable_length"] = true,
            [a + "audio.max_frames"] = (long)WindowSeconds * sampleRate / hop,
            [a + "audio.chunk_samples"] = WindowSeconds * sampleRate,
            [a + "decoder.type"] = head,
            [a + "decoder.blank_id"] = Int(g, "tokenizer.ggml.blank_token_id"),
            [a + "decoder.layer_count"] = Int(g, Parakeet + "predictor.n_layers", 1),
            [a + "decoder.max_symbols_per_step"] = Int(g, Parakeet + "tdt.max_symbols", 10),
            [a + "joint.activation"] = Text(g, Parakeet + "joint.activation", "relu"),
        };
        if (head == "tdt" && g.All.TryGetValue(Parakeet + "tdt.durations", out var durations) && durations is Array array)
        {
            result[a + "decoder.durations"] = array.Cast<object>().Select(d => Convert.ToInt64(d, System.Globalization.CultureInfo.InvariantCulture)).ToArray();
        }

        return result;
    }

    /// <summary>The voice-activity head of the "ultra" files (<c>vad.*</c>): recordings are segmented by the pipeline's own detector.</summary>
    public bool IgnoresTensor(string ggufName, GgufMetadataView g) => ggufName.StartsWith("vad.", StringComparison.Ordinal);

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, IWeightSource> SyntheticAttachments(GgufMetadataView g)
    {
        int bins = Int(g, Frontend + "num_mels", 128);
        int fft = Int(g, Frontend + "n_fft", 512);
        int rate = Int(g, Frontend + "sample_rate", 16000);
        return new Dictionary<string, IWeightSource>(StringComparer.Ordinal)
        {
            [MelFilters] = new MemoryWeightSource(() => MemoryMarshal.AsBytes(SlaneyMelFilterBank.Create(bins, fft, rate).AsSpan()).ToArray()),
        };
    }

    /// <inheritdoc/>
    public string? MapWeightName(string name, GgufMetadataView g)
    {
        if (s_modelTensors.TryGetValue(name, out var canonical))
        {
            return canonical;
        }

        var subsampling = Subsampling().Match(name);
        if (subsampling.Success)
        {
            int index = Array.IndexOf(s_subsamplingModules, int.Parse(subsampling.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture));
            return index < 0 ? null : $"encoder.subsampling.{index}.{subsampling.Groups[2].Value}";
        }

        var lstm = Lstm().Match(name);
        if (lstm.Success)
        {
            string layer = lstm.Groups[1].Value;
            return lstm.Groups[2].Value switch
            {
                "Wx" => $"decoder.lstm.{layer}.input.weight",
                "Wh" => $"decoder.lstm.{layer}.hidden.weight",
                _ => $"decoder.lstm.{layer}.input.bias",
            };
        }

        var bias = PositionBias().Match(name);
        if (bias.Success)
        {
            return $"encoder.layers.{bias.Groups[1].Value}.attention.bias_{bias.Groups[2].Value}";
        }

        var block = Block().Match(name);
        if (!block.Success)
        {
            return null;
        }

        string prefix = $"encoder.layers.{block.Groups[1].Value}.";
        string module = block.Groups[2].Value;
        string kind = block.Groups[3].Value;
        if (!s_blockModules.TryGetValue(module, out var mapped))
        {
            return null;
        }

        return prefix + mapped + "." + kind switch
        {
            "running_mean" => "mean",
            "running_var" => "variance",
            _ => kind,
        };
    }

    /// <inheritdoc/>
    public WeightInfo MapWeight(WeightInfo weight, GgufMetadataView g)
    {
        // Pointwise convolutions are linear layers: [O, C, 1] → [O, C].
        return weight.Name.Contains(".pointwise", StringComparison.Ordinal) && weight.Name.EndsWith(".weight", StringComparison.Ordinal) && weight.Shape.Rank == 3
            ? weight.As(weight.Name, new Shape(weight.Shape[0], weight.Shape[1]))
            : weight;
    }

    /// <summary>The blank is the last token; the vocabulary tokens are SentencePiece pieces.</summary>
    public void RefineHyperParameters(Dictionary<string, object> metadata, GgufMetadataView g, IReadOnlyDictionary<string, WeightInfo> weights)
    {
        metadata[CanonicalKeys.TokenizerModel] = "bpe";
        metadata[CanonicalKeys.TokenizerPre] = "default";
        metadata[CanonicalKeys.TokenizerAddSpacePrefix] = true;
        metadata[CanonicalKeys.TokenizerAddBos] = false;
    }

    private static int Int(GgufMetadataView g, string key)
        => g.All.TryGetValue(key, out var value) ? Convert.ToInt32(value, System.Globalization.CultureInfo.InvariantCulture) : throw new InvalidDataException($"parakeet GGUF: missing '{key}'.");

    private static int Int(GgufMetadataView g, string key, int fallback)
        => g.All.TryGetValue(key, out var value) ? Convert.ToInt32(value, System.Globalization.CultureInfo.InvariantCulture) : fallback;

    private static string Text(GgufMetadataView g, string key, string fallback) => g.All.TryGetValue(key, out var value) && value is string text ? text : fallback;

    [GeneratedRegex(@"^enc\.pre_encode\.conv\.(\d+)\.(weight|bias)$")]
    private static partial Regex Subsampling();

    [GeneratedRegex(@"^pred\.lstm\.(\d+)\.(Wx|Wh|bias)$")]
    private static partial Regex Lstm();

    [GeneratedRegex(@"^enc\.blocks\.(\d+)\.attn\.pos_bias_([uv])$")]
    private static partial Regex PositionBias();

    [GeneratedRegex(@"^enc\.blocks\.(\d+)\.(.+)\.(weight|bias|running_mean|running_var)$")]
    private static partial Regex Block();

    /// <summary>The subsampling modules that are convolutions, in the order of NeMo's module list.</summary>
    private static readonly int[] s_subsamplingModules = [0, 2, 3, 5, 6];

    private static readonly Dictionary<string, string> s_blockModules = new(StringComparer.Ordinal)
    {
        ["ff1.linear1"] = "ff1.up",
        ["ff1.linear2"] = "ff1.down",
        ["ff2.linear1"] = "ff2.up",
        ["ff2.linear2"] = "ff2.down",
        ["norm_ff1"] = "ff1_norm",
        ["norm_ff2"] = "ff2_norm",
        ["norm_attn"] = "attention_norm",
        ["norm_conv"] = "conv_norm",
        ["norm_out"] = "output_norm",
        ["attn.linear_q"] = "attention.q",
        ["attn.linear_k"] = "attention.k",
        ["attn.linear_v"] = "attention.v",
        ["attn.linear_out"] = "attention.o",
        ["attn.linear_pos"] = "attention.position",
        ["conv.pointwise1"] = "conv.pointwise1",
        ["conv.pointwise2"] = "conv.pointwise2",
        ["conv.depthwise"] = "conv.depthwise",
        ["conv.bn"] = "conv.norm",
    };

    private static readonly Dictionary<string, string> s_modelTensors = new(StringComparer.Ordinal)
    {
        ["enc.pre_encode.out.weight"] = "encoder.subsampling.out.weight",
        ["enc.pre_encode.out.bias"] = "encoder.subsampling.out.bias",
        ["pred.embed.weight"] = "decoder.embedding.weight",
        ["joint.enc.weight"] = "joint.encoder.weight",
        ["joint.enc.bias"] = "joint.encoder.bias",
        ["joint.pred.weight"] = "decoder.projection.weight",
        ["joint.pred.bias"] = "decoder.projection.bias",
        ["joint.out.weight"] = "joint.output.weight",
        ["joint.out.bias"] = "joint.output.bias",
    };
}
