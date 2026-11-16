using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Hugging Face Parakeet checkpoints (<c>parakeet_ctc</c>, <c>parakeet_tdt</c>, <c>parakeet_rnnt</c>: nvidia/parakeet-ctc-0.6b / 1.1b,
/// parakeet-tdt-0.6b-v3, parakeet-rnnt-0.6b in transformers format) → canonical <c>conformer_asr</c> (documented on
/// <c>ConformerAsrAdapter</c> and <c>ConformerEncoder</c>): NeMo's log-mel front end (pre-emphasis, a symmetric 400-sample window in a
/// 512-point FFT, ln(mel + 2⁻²⁴), per-channel normalization; the slaney filter bank as the <c>feature-extractor/mel-filters</c>
/// attachment), the depthwise-separable subsampling stack (a 3×3 convolution, then two depthwise 3×3 plus pointwise pairs, each stride 2),
/// conformer blocks with relative-position attention, and the CTC head or the LSTM prediction network and joint network of the
/// transducers (TDT: the joint's last logits are the durations). Shared by Canary's encoder.
/// </summary>
internal sealed partial class ParakeetHfMapping : IHfArchitectureMapping
{
    /// <summary>The longest window of the encoder, in seconds (longer recordings are transcribed window by window).</summary>
    internal const int WindowSeconds = 30;

    /// <summary>The subsampling convolutions in the order of transformers' module list (indices of the modules that are convolutions).</summary>
    private static readonly int[] s_subsamplingModules = [0, 2, 3, 5, 6];

    private static readonly Dictionary<string, string> s_layerModules = new(StringComparer.Ordinal)
    {
        ["feed_forward1.linear1"] = "ff1.up",
        ["feed_forward1.linear2"] = "ff1.down",
        ["feed_forward2.linear1"] = "ff2.up",
        ["feed_forward2.linear2"] = "ff2.down",
        ["norm_feed_forward1"] = "ff1_norm",
        ["norm_feed_forward2"] = "ff2_norm",
        ["norm_self_att"] = "attention_norm",
        ["norm_conv"] = "conv_norm",
        ["norm_out"] = "output_norm",
        ["self_attn.q_proj"] = "attention.q",
        ["self_attn.k_proj"] = "attention.k",
        ["self_attn.v_proj"] = "attention.v",
        ["self_attn.o_proj"] = "attention.o",
        ["self_attn.relative_k_proj"] = "attention.position",
        ["conv.pointwise_conv1"] = "conv.pointwise1",
        ["conv.pointwise_conv2"] = "conv.pointwise2",
        ["conv.depthwise_conv"] = "conv.depthwise",
        ["conv.norm"] = "conv.norm",
    };

    /// <inheritdoc/>
    public string Architecture => "conformer_asr";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["parakeet_ctc", "parakeet_tdt", "parakeet_rnnt"];

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig config) => ValidateEncoder(config, "parakeet");

    /// <inheritdoc/>
    public string? PartialSupport(string modelType) => modelType == "parakeet_rnnt" ? "RNN-T checkpoints map like TDT ones but are not verified" : null;

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        string a = Architecture + ".";
        var result = EncoderHyperParameters(c, a);
        string type = c.ModelType switch
        {
            "parakeet_ctc" => "ctc",
            "parakeet_tdt" => "tdt",
            _ => "rnnt",
        };
        result[a + "decoder.type"] = type;
        result[a + "decoder.blank_id"] = type == "ctc" ? c.GetInt("pad_token_id") ?? c.Int("vocab_size") - 1 : c.Int("blank_token_id");
        if (type != "ctc")
        {
            result[a + "decoder.layer_count"] = c.GetInt("num_decoder_layers") ?? 1;
            result[a + "decoder.max_symbols_per_step"] = c.GetInt("max_symbols_per_step") ?? 10;
            result[a + "joint.activation"] = c.GetString("hidden_act") ?? "relu";
            if (c.Get("durations") is { ValueKind: JsonValueKind.Array } durations)
            {
                result[a + "decoder.durations"] = durations.EnumerateArray().Select(d => d.GetInt64()).ToArray();
            }
        }

        return result;
    }

    [GeneratedRegex(@"^encoder\.layers\.(\d+)\.(.+)\.(weight|bias|bias_u|bias_v|running_mean|running_var|num_batches_tracked)$")]
    private static partial Regex LayerName();

    [GeneratedRegex(@"^encoder\.subsampling\.layers\.(\d+)\.(weight|bias)$")]
    private static partial Regex SubsamplingName();

    [GeneratedRegex(@"^decoder\.lstm\.(weight|bias)_(ih|hh)_l(\d+)$")]
    private static partial Regex LstmName();

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        var collector = new HfWeightCollector();
        foreach (var (name, tensor) in tensors)
        {
            var info = tensor.ToWeightInfo();
            if (TryMapEncoder(name, info, collector))
            {
                continue;
            }

            switch (name)
            {
                case "ctc_head.weight":
                    collector.Add(info.As("ctc.weight", new Tensors.Shape(info.Shape[0], info.Shape[1])));
                    continue;
                case "ctc_head.bias":
                    collector.Add(HfTransforms.Plain(info, "ctc.bias"));
                    continue;
                case "encoder_projector.weight" or "encoder_projector.bias":
                    collector.Add(HfTransforms.Plain(info, "joint.encoder." + Suffix(name)));
                    continue;
                case "joint.head.weight" or "joint.head.bias":
                    collector.Add(HfTransforms.Plain(info, "joint.output." + Suffix(name)));
                    continue;
                case "decoder.embedding.weight":
                    collector.Add(info.As("decoder.embedding.weight"));
                    continue;
                case "decoder.decoder_projector.weight" or "decoder.decoder_projector.bias":
                    collector.Add(HfTransforms.Plain(info, "decoder.projection." + Suffix(name)));
                    continue;
            }

            var lstm = LstmName().Match(name);
            if (lstm.Success)
            {
                string gate = lstm.Groups[2].Value == "ih" ? "input" : "hidden";
                collector.Add(HfTransforms.Plain(info, $"decoder.lstm.{lstm.Groups[3].Value}.{gate}.{lstm.Groups[1].Value}"));
                continue;
            }

            throw new NotSupportedException($"{c.ModelType}: tensor '{name}' has no canonical mapping.");
        }

        return collector.Weights;
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, IWeightSource> MapAttachments(HfConfig config, string directory) => FilterBank(config);

    /// <summary>The reasons a FastConformer encoder configuration is not supported.</summary>
    /// <param name="config">The checkpoint's configuration.</param>
    /// <param name="family">The family named in the reasons.</param>
    internal static IEnumerable<string> ValidateEncoder(HfConfig config, string family)
    {
        if (!config.Root.TryGetProperty("encoder_config", out var encoder) || encoder.ValueKind != JsonValueKind.Object)
        {
            yield return $"{family}: no encoder_config.";
            yield break;
        }

        int heads = Int(encoder, "num_attention_heads", 8);
        if (Int(encoder, "num_key_value_heads", heads) != heads)
        {
            yield return $"{family}: grouped key-value heads are not supported.";
        }

        if (Int(encoder, "subsampling_factor", 8) != 8 || Int(encoder, "subsampling_conv_stride", 2) != 2 || Int(encoder, "subsampling_conv_kernel_size", 3) != 3)
        {
            yield return $"{family}: only the 8× depthwise-separable subsampling with 3×3 kernels is supported.";
        }
    }

    /// <summary>The canonical encoder and audio hyper-parameters of a FastConformer <c>encoder_config</c> under the prefix <paramref name="a"/>.</summary>
    /// <param name="c">The checkpoint's configuration.</param>
    /// <param name="a">The canonical prefix, e.g. <c>conformer_asr.</c>.</param>
    internal static Dictionary<string, object> EncoderHyperParameters(HfConfig c, string a)
    {
        var encoder = c.Root.GetProperty("encoder_config");
        int hidden = Int(encoder, "hidden_size", 1024);
        int channels = Int(encoder, "subsampling_conv_channels", 256);
        var audio = Preprocessor(c);
        int sampleRate = audio.SampleRate;
        return new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "hidden_size"] = hidden,
            [a + "encoder.layer_count"] = Int(encoder, "num_hidden_layers", 17),
            [a + "encoder.attention.head_count"] = Int(encoder, "num_attention_heads", 8),
            [a + "encoder.subsampling.strides"] = new long[] { 2, 2, 1, 2, 1 },
            [a + "encoder.subsampling.groups"] = new long[] { 1, channels, 1, channels, 1 },
            [a + "encoder.subsampling.activate"] = new[] { true, false, true, false, true },
            [a + "encoder.input_scale"] = encoder.TryGetProperty("scale_input", out var scale) && scale.ValueKind == JsonValueKind.True ? MathF.Sqrt(hidden) : 1f,
            [a + "encoder.mlp.activation"] = encoder.TryGetProperty("hidden_act", out var act) && act.GetString() is { } name ? name : "silu",
            [a + "encoder.norm_eps"] = 1e-5f,
            [a + "encoder.conv.norm_eps"] = 1e-5f,
            [a + "audio.features"] = "log-mel",
            [a + "audio.sample_rate"] = sampleRate,
            [a + "audio.fft_size"] = audio.FftSize,
            [a + "audio.window_size"] = audio.WindowSize,
            [a + "audio.periodic_window"] = false,
            [a + "audio.hop_length"] = audio.HopLength,
            [a + "audio.mel_bins"] = audio.MelBins,
            [a + "audio.preemphasis"] = audio.Preemphasis,
            [a + "audio.pad_mode"] = "constant",
            [a + "audio.log_base"] = "e",
            [a + "audio.log_guard"] = Math.Pow(2, -24),
            [a + "audio.dynamic_range"] = 0.0,
            [a + "audio.log_offset"] = 0.0,
            [a + "audio.log_scale"] = 1.0,
            [a + "audio.mel_normalize"] = "per-channel",
            [a + "audio.variable_length"] = true,
            [a + "audio.max_frames"] = (long)WindowSeconds * sampleRate / audio.HopLength,
            [a + "audio.chunk_samples"] = WindowSeconds * sampleRate,
        };
    }

    /// <summary>Maps a FastConformer encoder tensor of transformers' naming; false when <paramref name="name"/> is not one.</summary>
    /// <param name="name">The checkpoint tensor name.</param>
    /// <param name="info">The tensor.</param>
    /// <param name="collector">Receives the canonical weight.</param>
    internal static bool TryMapEncoder(string name, WeightInfo info, HfWeightCollector collector)
    {
        switch (name)
        {
            case "encoder.subsampling.linear.weight" or "encoder.subsampling.linear.bias":
                collector.Add(HfTransforms.Plain(info, "encoder.subsampling.out." + Suffix(name)));
                return true;
        }

        var subsampling = SubsamplingName().Match(name);
        if (subsampling.Success)
        {
            int index = Array.IndexOf(s_subsamplingModules, int.Parse(subsampling.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture));
            if (index < 0)
            {
                throw new NotSupportedException($"FastConformer: unexpected subsampling tensor '{name}'.");
            }

            collector.Add(HfTransforms.Plain(info, $"encoder.subsampling.{index}.{subsampling.Groups[2].Value}"));
            return true;
        }

        var layer = LayerName().Match(name);
        if (!layer.Success)
        {
            return false;
        }

        string prefix = $"encoder.layers.{layer.Groups[1].Value}.";
        string module = layer.Groups[2].Value;
        string kind = layer.Groups[3].Value;
        if (kind == "num_batches_tracked")
        {
            return true;
        }

        if (module == "self_attn" && kind is "bias_u" or "bias_v")
        {
            collector.Add(HfTransforms.Plain(info, prefix + "attention." + kind));
            return true;
        }

        if (!s_layerModules.TryGetValue(module, out var canonical))
        {
            throw new NotSupportedException($"FastConformer: tensor '{name}' has no canonical mapping.");
        }

        string target = kind switch
        {
            "running_mean" => "mean",
            "running_var" => "variance",
            _ => kind,
        };
        // Pointwise convolutions are linear layers: [O, C, 1] → [O, C].
        var mapped = info.Shape.Rank == 3 && module.Contains("pointwise", StringComparison.Ordinal)
            ? info.As(prefix + canonical + "." + target, new Tensors.Shape(info.Shape[0], info.Shape[1]))
            : HfTransforms.Plain(info, prefix + canonical + "." + target);
        collector.Add(mapped);
        return true;
    }

    /// <summary>The slaney filter bank of the checkpoint's front end as the <c>feature-extractor/mel-filters</c> attachment.</summary>
    /// <param name="config">The checkpoint's configuration.</param>
    internal static IReadOnlyDictionary<string, IWeightSource> FilterBank(HfConfig config)
    {
        var audio = Preprocessor(config);
        return new Dictionary<string, IWeightSource>(StringComparer.Ordinal)
        {
            ["feature-extractor/mel-filters"] = new MemoryWeightSource(
                () => MemoryMarshal.AsBytes(HfMelFilterBank.Create(audio.MelBins, audio.FftSize, audio.SampleRate).AsSpan()).ToArray()),
        };
    }

    private static string Suffix(string name) => name[(name.LastIndexOf('.') + 1)..];

    private static int Int(JsonElement element, string key, int fallback)
        => element.TryGetProperty(key, out var value) && value.ValueKind == JsonValueKind.Number ? value.GetInt32() : fallback;

    // The feature extractor of preprocessor_config.json or processor_config.json (transformers' defaults when absent).
    private static AudioSettings Preprocessor(HfConfig config)
    {
        JsonElement? root = null;
        foreach (string file in (string[])["preprocessor_config.json", "processor_config.json"])
        {
            string path = Path.Combine(config.Directory, file);
            if (File.Exists(path))
            {
                var document = JsonDocument.Parse(File.ReadAllBytes(path)).RootElement.Clone();
                root = document.TryGetProperty("feature_extractor", out var nested) ? nested : document;
                break;
            }
        }

        int Value(string key, int fallback) => root is { } r && r.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : fallback;
        double preemphasis = root is { } p && p.TryGetProperty("preemphasis", out var e) && e.ValueKind == JsonValueKind.Number ? e.GetDouble() : 0.97;
        int mels = Value("feature_size", config.Root.TryGetProperty("encoder_config", out var encoder) ? Int(encoder, "num_mel_bins", 80) : 80);
        return new AudioSettings(Value("sampling_rate", 16000), Value("n_fft", 512), Value("win_length", 400), Value("hop_length", 160), mels, preemphasis);
    }

    private readonly record struct AudioSettings(int SampleRate, int FftSize, int WindowSize, int HopLength, int MelBins, double Preemphasis);
}
