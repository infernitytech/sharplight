using System.Runtime.InteropServices;
using System.Text.Json;
using Sharplight.Audio;
using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Voxtral (HF <c>voxtral</c>: Voxtral Mini 3B, Voxtral Small 24B) → canonical <c>decoder</c> with a Whisper-encoder audio tower: the Llama /
/// Mistral language model is mapped by <see cref="DecoderHfMapping"/>, the Whisper large-v3 encoder to the core's "whisper" audio tower
/// (<c>audio.*</c>: two convolutions, learned positions, layers whose key projection has no bias, the final LayerNorm) with the projector that
/// concatenates every four rows, together with Whisper's log-mel front end and mistral-common's transcription prompt
/// (<c>&lt;s&gt;[INST][BEGIN_AUDIO]</c> audio <c>[/INST]lang:{code}[TRANSCRIBE]</c>). The tokenizer is the checkpoint's <c>tekken.json</c>.
/// </summary>
internal sealed class VoxtralHfMapping : IHfArchitectureMapping
{
    private readonly DecoderHfMapping _text = new();

    /// <inheritdoc/>
    public string Architecture => _text.Architecture;

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["voxtral"];

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig config, HuggingFaceImportOptions options)
    {
        var md = new Dictionary<string, object>(_text.MapHyperParameters(config, options), StringComparer.Ordinal);
        var audio = AudioConfig(config);
        var features = Features(config.Directory);
        string a = Architecture + ".audio.";
        int hidden = Int(audio, "hidden_size", 1280);
        md[a + "tower"] = "whisper";
        md[a + "layer_count"] = Int(audio, "num_hidden_layers", 32);
        md[a + "hidden_size"] = hidden;
        md[a + "head_count"] = Int(audio, "num_attention_heads", 20);
        md[a + "activation"] = config.Root.TryGetProperty("projector_hidden_act", out var act) && act.GetString() is { } name ? name : "gelu";
        md[a + "downsample"] = Int(audio, "intermediate_size", 4 * hidden) / hidden;
        md[a + "mel_bins"] = Int(audio, "num_mel_bins", 128);
        md[a + "window_frames"] = 2 * Int(audio, "max_source_positions", 1500);
        md[a + "sample_rate"] = features.Rate;
        md[a + "fft_size"] = features.Fft;
        md[a + "hop_length"] = features.Hop;
        md[a + "chunk_samples"] = features.Samples;
        md[a + "token_id"] = config.Root.TryGetProperty("audio_token_id", out var token) ? token.GetInt32() : 24;
        md[a + "prompt.template"] = "<s>[INST][BEGIN_AUDIO]{audio}[/INST]{language}[TRANSCRIBE]";
        md[a + "prompt.language_prefix"] = "lang:{code}";
        return md;
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig config, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        var text = new Dictionary<string, SafetensorsTensor>(StringComparer.Ordinal);
        var weights = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
        foreach (var (name, tensor) in tensors)
        {
            if (name.StartsWith("language_model.", StringComparison.Ordinal))
            {
                text[name["language_model.".Length..]] = tensor;
                continue;
            }

            string canonical = AudioName(name) ?? throw new NotSupportedException($"voxtral: tensor '{name}' has no canonical mapping.");
            var info = tensor.ToWeightInfo();
            var w = canonical == "audio.position_embedding.weight" ? HfTransforms.ToF32(info, canonical)
                : info.Shape.Rank == 1 ? HfTransforms.Plain(info, canonical) : info.As(canonical);
            weights[w.Name] = w;
        }

        foreach (var (name, w) in _text.MapWeights(config, text, options))
        {
            weights[name] = w;
        }

        return weights;
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, IWeightSource> MapAttachments(HfConfig config, string directory)
    {
        int mels = Int(AudioConfig(config), "num_mel_bins", 128);
        var features = Features(directory);
        return new Dictionary<string, IWeightSource>(StringComparer.Ordinal)
        {
            ["feature-extractor/mel-filters"] = new MemoryWeightSource(
                () => MemoryMarshal.AsBytes(SlaneyMelFilterBank.Create(mels, features.Fft, features.Rate).AsSpan()).ToArray()),
        };
    }

    private static string? AudioName(string name)
    {
        const string Tower = "audio_tower.";
        if (name.StartsWith("multi_modal_projector.linear_", StringComparison.Ordinal))
        {
            return "audio.projector." + name["multi_modal_projector.linear_".Length..];
        }

        if (!name.StartsWith(Tower, StringComparison.Ordinal))
        {
            return null;
        }

        string local = name[Tower.Length..];
        string[] parts = local.Split('.');
        string kind = parts[^1];
        string module = string.Join('.', parts[..^1]);
        string? canonical = module switch
        {
            "conv1" => "conv.0",
            "conv2" => "conv.1",
            "embed_positions" => "position_embedding",
            "layer_norm" => "output_norm",
            _ when parts.Length == 4 && parts[0] == "layers" => parts[2] switch
            {
                "self_attn_layer_norm" => $"layers.{parts[1]}.input_norm",
                "final_layer_norm" => $"layers.{parts[1]}.post_attention_norm",
                "fc1" => $"layers.{parts[1]}.mlp.up",
                "fc2" => $"layers.{parts[1]}.mlp.down",
                _ => null,
            },
            _ when parts.Length == 5 && parts[0] == "layers" && parts[2] == "self_attn" => parts[3] switch
            {
                "q_proj" => $"layers.{parts[1]}.attention.q",
                "k_proj" => $"layers.{parts[1]}.attention.k",
                "v_proj" => $"layers.{parts[1]}.attention.v",
                "out_proj" => $"layers.{parts[1]}.attention.o",
                _ => null,
            },
            _ => null,
        };
        return canonical is null ? null : $"audio.{canonical}.{kind}";
    }

    private static JsonElement AudioConfig(HfConfig config)
        => config.Root.TryGetProperty("audio_config", out var audio) ? audio : throw new NotSupportedException("voxtral: config.json has no audio_config.");

    private static int Int(JsonElement e, string key, int fallback)
        => e.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : fallback;

    private static (int Rate, int Fft, int Hop, int Samples) Features(string directory)
    {
        string path = Path.Combine(directory, "preprocessor_config.json");
        if (!File.Exists(path))
        {
            return (16000, 400, 160, 480000);
        }

        using var document = JsonDocument.Parse(File.ReadAllText(path));
        var root = document.RootElement;
        int rate = Int(root, "sampling_rate", 16000);
        return (rate, Int(root, "n_fft", 400), Int(root, "hop_length", 160), Int(root, "n_samples", 30 * rate));
    }
}
