using System.Runtime.InteropServices;
using System.Text.Json;
using Sharplight.Audio;
using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Qwen3-ASR (HF <c>qwen3_asr</c>) → canonical <c>decoder</c> with an audio tower: the Qwen3 language model is mapped by
/// <see cref="DecoderHfMapping"/>, the Whisper-derived audio tower (three strided 3×3 convolutions per 100-frame chunk, windowed attention
/// layers, the two-layer projector) to the <c>audio.*</c> weights and <c>decoder.audio.*</c> keys of the core's audio tower, together with the
/// log-mel front end (Whisper's, 128 bins) and the transcription prompt ("language {name}&lt;asr_text&gt;" forces the language). Both
/// published layouts import: the original release (<c>thinker.*</c> tensors, <c>thinker_config</c>, the projector inside the tower as
/// <c>proj1</c> / <c>proj2</c>) and transformers' <c>-hf</c> conversion (<c>model.audio_tower.*</c>, <c>model.multi_modal_projector.*</c>,
/// <c>model.language_model.*</c>).
/// </summary>
internal sealed class Qwen3AsrHfMapping : IHfArchitectureMapping
{
    /// <summary>The transcription prompt of the chat template (empty system turn unless the request gives context).</summary>
    internal const string PromptTemplate = "<|im_start|>system\n{context}<|im_end|>\n<|im_start|>user\n<|audio_start|>{audio}<|audio_end|><|im_end|>\n<|im_start|>assistant\n";

    /// <summary>The assistant prefix that forces the language.</summary>
    internal const string LanguagePrefix = "language {language}<asr_text>";

    /// <summary>The marker between the language preamble and the transcript.</summary>
    internal const string TranscriptMarker = "<asr_text>";

    private readonly DecoderHfMapping _text = new();

    /// <inheritdoc/>
    public string Architecture => _text.Architecture;

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["qwen3_asr"];

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig config, HuggingFaceImportOptions options)
    {
        var md = new Dictionary<string, object>(_text.MapHyperParameters(config, options), StringComparer.Ordinal);
        var audio = AudioConfig(config);
        var features = FeatureSettings.Read(config.Directory);
        string a = Architecture + ".audio.";
        int nWindow = Int(audio, "n_window", 50);
        md[a + "tower"] = "chunked-conv2d";
        md[a + "layer_count"] = Int(audio, "encoder_layers", 0);
        md[a + "hidden_size"] = Int(audio, "d_model", 0);
        md[a + "head_count"] = Int(audio, "encoder_attention_heads", 0);
        md[a + "activation"] = String(audio, "activation_function", "gelu");
        md[a + "chunk_frames"] = 2 * nWindow;
        md[a + "window_frames"] = Int(audio, "n_window_infer", 800);
        md[a + "mel_bins"] = Int(audio, "num_mel_bins", 128);
        md[a + "sample_rate"] = features.SampleRate;
        md[a + "fft_size"] = features.FftSize;
        md[a + "hop_length"] = features.HopLength;
        md[a + "chunk_samples"] = features.ChunkSamples;
        md[a + "min_samples"] = features.MinSamples;
        md[a + "token_id"] = TokenId(config, "audio_token_id");
        md[a + "prompt.template"] = PromptTemplate;
        md[a + "prompt.language_prefix"] = LanguagePrefix;
        md[a + "prompt.transcript_marker"] = TranscriptMarker;
        return md;
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig config, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        var text = new Dictionary<string, SafetensorsTensor>(StringComparer.Ordinal);
        var audio = new List<(string Name, SafetensorsTensor Tensor)>();
        foreach (var (name, tensor) in tensors)
        {
            if (AudioName(name) is { } tower)
            {
                audio.Add((tower, tensor));
            }
            else if (TextName(name) is { } textName)
            {
                text[textName] = tensor;
            }
            else
            {
                throw new NotSupportedException($"qwen3_asr: tensor '{name}' has no canonical mapping.");
            }
        }

        var weights = new Dictionary<string, WeightInfo>(_text.MapWeights(config, text, options), StringComparer.Ordinal);
        foreach (var (name, tensor) in audio)
        {
            var info = tensor.ToWeightInfo();
            var w = info.Shape.Rank == 1 ? HfTransforms.Plain(info, name) : info.As(name);
            weights[w.Name] = w;
        }

        return weights;
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, IWeightSource> MapAttachments(HfConfig config, string directory)
    {
        var audio = AudioConfig(config);
        int mels = Int(audio, "num_mel_bins", 128);
        var features = FeatureSettings.Read(directory);
        return new Dictionary<string, IWeightSource>(StringComparer.Ordinal)
        {
            ["feature-extractor/mel-filters"] = new MemoryWeightSource(
                () => MemoryMarshal.AsBytes(SlaneyMelFilterBank.Create(mels, features.FftSize, features.SampleRate).AsSpan()).ToArray()),
        };
    }

    // The language model's tensors under the plain Qwen3 names (model.*, lm_head.weight).
    private static string? TextName(string name)
    {
        foreach (string prefix in (string[])["thinker.model.", "model.language_model."])
        {
            if (name.StartsWith(prefix, StringComparison.Ordinal))
            {
                return "model." + name[prefix.Length..];
            }
        }

        return name is "thinker.lm_head.weight" or "lm_head.weight" ? "lm_head.weight" : null;
    }

    // The audio tower's and projector's tensors under the canonical audio.* names, or null for a language-model tensor.
    private static string? AudioName(string name)
    {
        string? local = null;
        foreach (string prefix in (string[])["thinker.audio_tower.", "model.audio_tower."])
        {
            if (name.StartsWith(prefix, StringComparison.Ordinal))
            {
                local = name[prefix.Length..];
            }
        }

        if (name.StartsWith("model.multi_modal_projector.", StringComparison.Ordinal))
        {
            local = name["model.multi_modal_projector.".Length..].Replace("linear_", "proj", StringComparison.Ordinal);
        }

        if (local is null)
        {
            return null;
        }

        string[] parts = local.Split('.');
        string module = string.Join('.', parts[..^1]);
        string kind = parts[^1];
        string? canonical = module switch
        {
            "conv2d1" => "conv.0",
            "conv2d2" => "conv.1",
            "conv2d3" => "conv.2",
            "conv_out" => "conv_out",
            "ln_post" => "output_norm",
            "proj1" => "projector.1",
            "proj2" => "projector.2",
            _ when parts.Length == 4 && parts[0] == "layers" => $"layers.{parts[1]}.{LayerModule(parts[2])}",
            _ when parts.Length == 5 && parts[0] == "layers" && parts[2] == "self_attn" => $"layers.{parts[1]}.attention.{AttentionModule(parts[3])}",
            _ => null,
        };
        return canonical is null || canonical.EndsWith(".?", StringComparison.Ordinal)
            ? throw new NotSupportedException($"qwen3_asr: audio tensor '{name}' has no canonical mapping.")
            : $"audio.{canonical}.{kind}";
    }

    private static string LayerModule(string module) => module switch
    {
        "self_attn_layer_norm" => "input_norm",
        "final_layer_norm" => "post_attention_norm",
        "fc1" => "mlp.up",
        "fc2" => "mlp.down",
        _ => "?",
    };

    private static string AttentionModule(string module) => module switch
    {
        "q_proj" => "q",
        "k_proj" => "k",
        "v_proj" => "v",
        "out_proj" => "o",
        _ => "?",
    };

    private static JsonElement AudioConfig(HfConfig config)
    {
        var root = config.Root;
        if (root.TryGetProperty("thinker_config", out var thinker) && thinker.TryGetProperty("audio_config", out var nested))
        {
            return nested;
        }

        return root.TryGetProperty("audio_config", out var audio) ? audio : throw new NotSupportedException("qwen3_asr: config.json has no audio_config.");
    }

    private static int TokenId(HfConfig config, string key)
    {
        var root = config.Root;
        if (root.TryGetProperty(key, out var id) && id.ValueKind == JsonValueKind.Number)
        {
            return id.GetInt32();
        }

        return root.TryGetProperty("thinker_config", out var thinker) && thinker.TryGetProperty(key, out id) && id.ValueKind == JsonValueKind.Number
            ? id.GetInt32()
            : throw new NotSupportedException($"qwen3_asr: config.json has no {key}.");
    }

    private static int Int(JsonElement e, string key, int fallback)
        => e.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : fallback;

    private static string String(JsonElement e, string key, string fallback)
        => e.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() ?? fallback : fallback;

    /// <summary>The log-mel settings of the checkpoint's preprocessor (<c>preprocessor_config.json</c>, or the <c>feature_extractor</c> of <c>processor_config.json</c>).</summary>
    private readonly record struct FeatureSettings(int SampleRate, int FftSize, int HopLength, int ChunkSamples, int MinSamples)
    {
        public static FeatureSettings Read(string directory)
        {
            JsonElement? settings = null;
            using var preprocessor = ReadJson(directory, "preprocessor_config.json");
            using var processor = ReadJson(directory, "processor_config.json");
            if (preprocessor is not null)
            {
                settings = preprocessor.RootElement;
            }
            else if (processor is not null && processor.RootElement.TryGetProperty("feature_extractor", out var nested))
            {
                settings = nested;
            }

            int rate = settings is { } s1 ? Int(s1, "sampling_rate", 16000) : 16000;
            int samples = settings is { } s2 ? Int(s2, "n_samples", 0) : 0;
            if (samples == 0)
            {
                samples = (settings is { } s3 ? Int(s3, "chunk_length", 30) : 30) * rate;
            }

            return new FeatureSettings(rate, settings is { } s4 ? Int(s4, "n_fft", 400) : 400, settings is { } s5 ? Int(s5, "hop_length", 160) : 160, samples,
                settings is { } s6 ? Int(s6, "min_length", 8000) : 8000);
        }

        private static JsonDocument? ReadJson(string directory, string file)
        {
            string path = Path.Combine(directory, file);
            return File.Exists(path) ? JsonDocument.Parse(File.ReadAllText(path)) : null;
        }
    }
}
