using System.Runtime.InteropServices;
using System.Text.Json;
using Sharplight.Audio;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Voxtral Realtime (HF <c>voxtral_realtime</c>: Voxtral-Mini-4B-Realtime-2602) → canonical <c>decoder</c> with the causal streaming audio tower
/// (<c>audio.tower</c> "causal": causally padded convolutions, rotary RMS-norm layers with a sliding window, rows concatenated four at a time and
/// projected). The Mistral language model is mapped by <see cref="DecoderHfMapping"/>; every decoder step adds the audio row of its position to the
/// token embedding (<c>audio.additive</c>). The model's time conditioning is a constant (the sinusoidal embedding of the number of delay tokens), so
/// the per-layer adaptive RMS-norm scale <c>1 + linear2(gelu(linear1(t)))</c> is folded into the post-attention norm weights here; the decoder is then
/// an ordinary Mistral model. The tokenizer is the checkpoint's <c>tekken.json</c>.
/// </summary>
internal sealed class VoxtralRealtimeHfMapping : IHfArchitectureMapping
{
    private readonly DecoderHfMapping _text = new();

    /// <inheritdoc/>
    public string Architecture => _text.Architecture;

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["voxtral_realtime"];

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig config, HuggingFaceImportOptions options)
    {
        var md = new Dictionary<string, object>(_text.MapHyperParameters(config, options), StringComparer.Ordinal);
        var audio = AudioConfig(config);
        var features = Features(config.Directory);
        string a = Architecture + ".audio.";
        int hidden = Int(audio, "hidden_size", 1280);
        int heads = Int(audio, "num_attention_heads", 32);
        md[a + "tower"] = "causal";
        md[a + "layer_count"] = Int(audio, "num_hidden_layers", 32);
        md[a + "hidden_size"] = hidden;
        md[a + "head_count"] = heads;
        md[a + "head_size"] = Int(audio, "head_dim", hidden / heads);
        md[a + "norm_eps"] = audio.TryGetProperty("rms_norm_eps", out var eps) ? eps.GetSingle() : 1e-5f;
        md[a + "sliding_window"] = Int(audio, "sliding_window", 0);
        md[a + "rope.base"] = audio.TryGetProperty("rope_parameters", out var rope) && rope.TryGetProperty("rope_theta", out var theta) ? theta.GetSingle() : 1e6f;
        md[a + "activation"] = config.Root.TryGetProperty("projector_hidden_act", out var act) && act.GetString() is { } name ? name : "gelu";
        md[a + "downsample"] = Int(config.Root, "downsample_factor", 4);
        md[a + "mel_bins"] = Int(audio, "num_mel_bins", 128);
        md[a + "window_frames"] = 6000;
        md[a + "sample_rate"] = features.Rate;
        md[a + "fft_size"] = features.Fft;
        md[a + "hop_length"] = features.Hop;
        md[a + "chunk_samples"] = 0;
        md[a + "log_peak"] = features.LogMax;
        md[a + "token_id"] = StreamingPadToken;
        md[a + "additive"] = true;
        int delay = Int(config.Root, "default_num_delay_tokens", 6);
        md[a + "stream.frames_per_token"] = Int(config.Root, "audio_length_per_tok", 8);
        md[a + "stream.delay_tokens"] = delay;
        md[a + "stream.left_pad_tokens"] = LeftPadTokens(config.Directory);
        md[a + "stream.right_pad_tokens"] = delay + 1 + OfflineBufferTokens;
        md[a + "prompt.template"] = "<s>{audio}";
        return md;
    }

    /// <summary>Extra silent steps after the audio of an offline request (mistral-common's <c>OFFLINE_STREAMING_BUFFER_TOKENS</c>).</summary>
    private const int OfflineBufferTokens = 10;

    private static int LeftPadTokens(string directory)
    {
        string path = Path.Combine(directory, "tekken.json");
        if (File.Exists(path))
        {
            using var stream = File.OpenRead(path);
            using var document = JsonDocument.Parse(stream);
            if (document.RootElement.TryGetProperty("audio", out var audio) && audio.TryGetProperty("streaming_n_left_pad_tokens", out var n) && n.ValueKind == JsonValueKind.Number)
            {
                return n.GetInt32();
            }
        }

        return 32;
    }

    /// <summary>The id of the <c>[STREAMING_PAD]</c> token in <c>tekken.json</c>.</summary>
    private const int StreamingPadToken = 32;

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig config, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        var text = new Dictionary<string, SafetensorsTensor>(StringComparer.Ordinal);
        var weights = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
        var ada = new Dictionary<string, SafetensorsTensor>(StringComparer.Ordinal);
        foreach (var (name, tensor) in tensors)
        {
            if (name.StartsWith("language_model.", StringComparison.Ordinal))
            {
                string local = name["language_model.".Length..];
                if (local.Contains(".ada_rms_norm.", StringComparison.Ordinal))
                {
                    ada[local] = tensor;
                }
                else
                {
                    text[local] = tensor;
                }

                continue;
            }

            string canonical = AudioName(name) ?? throw new NotSupportedException($"voxtral_realtime: tensor '{name}' has no canonical mapping.");
            var info = tensor.ToWeightInfo();
            var w = info.Shape.Rank == 1 ? HfTransforms.Plain(info, canonical) : info.As(canonical);
            weights[w.Name] = w;
        }

        var mapped = _text.MapWeights(config, text, options);
        foreach (var (name, w) in mapped)
        {
            weights[name] = w;
        }

        int delay = Int(config.Root, "default_num_delay_tokens", 6);
        int layers = config.Int("num_hidden_layers");
        for (int il = 0; il < layers; il++)
        {
            string target = $"layers.{il}.post_attention_norm.weight";
            string prefix = $"model.layers.{il}.ada_rms_norm.";
            var norm = mapped[target];
            var linear1 = ada[prefix + "linear1.weight"].ToWeightInfo();
            var linear2 = ada[prefix + "linear2.weight"].ToWeightInfo();
            weights[target] = WeightInfo.Derived(target, DataType.F32, norm.Shape, () => FoldedNorm(norm, linear1, linear2, delay));
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

    // norm · (1 + linear2(gelu(linear1(t)))) with t the sinusoidal embedding of the delay: [cos(delay · f_i)..., sin(delay · f_i)...].
    private static byte[] FoldedNorm(WeightInfo norm, WeightInfo linear1, WeightInfo linear2, int delay)
    {
        var scale = HfTransforms.DecodeF32(norm);
        int hidden = scale.Length;
        int half = hidden / 2;
        var t = new float[hidden];
        for (int i = 0; i < half; i++)
        {
            float frequency = MathF.Exp(-MathF.Log(10000f) * i / half);
            t[i] = MathF.Cos(delay * frequency);
            t[half + i] = MathF.Sin(delay * frequency);
        }

        var w1 = HfTransforms.DecodeF32(linear1);
        var w2 = HfTransforms.DecodeF32(linear2);
        int bottleneck = w1.Length / hidden;
        var mid = new float[bottleneck];
        for (int j = 0; j < bottleneck; j++)
        {
            double sum = 0;
            for (int i = 0; i < hidden; i++)
            {
                sum += w1[(j * hidden) + i] * (double)t[i];
            }

            mid[j] = (float)(0.5 * sum * (1.0 + Erf(sum / Math.Sqrt(2.0))));
        }

        var folded = new float[hidden];
        for (int i = 0; i < hidden; i++)
        {
            double sum = 0;
            for (int j = 0; j < bottleneck; j++)
            {
                sum += w2[(i * bottleneck) + j] * (double)mid[j];
            }

            folded[i] = (float)(scale[i] * (1.0 + sum));
        }

        return HfTransforms.EncodeF32(folded);
    }

    // erf to double precision (Abramowitz-Stegun 7.1.26 is too coarse here): series for small |x|, continued fraction otherwise.
    private static double Erf(double x)
    {
        double ax = Math.Abs(x);
        double result;
        if (ax < 2.5)
        {
            double term = ax;
            double sum = ax;
            for (int n = 1; n < 100; n++)
            {
                term *= -ax * ax / n;
                double add = term / ((2 * n) + 1);
                sum += add;
                if (Math.Abs(add) < 1e-17)
                {
                    break;
                }
            }

            result = 2.0 / Math.Sqrt(Math.PI) * sum;
        }
        else
        {
            double f = 0;
            for (int k = 60; k >= 1; k--)
            {
                f = k / 2.0 / (ax + f);
            }

            result = 1.0 - (Math.Exp(-ax * ax) / Math.Sqrt(Math.PI) / (ax + f));
        }

        return x < 0 ? -result : result;
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
            "embedder.conv1" => "conv.0",
            "embedder.conv2" => "conv.1",
            "norm" => "output_norm",
            _ when parts.Length == 4 && parts[0] == "layers" => parts[2] switch
            {
                "self_attn_layer_norm" => $"layers.{parts[1]}.input_norm",
                "final_layer_norm" => $"layers.{parts[1]}.post_attention_norm",
                _ => null,
            },
            _ when parts.Length == 5 && parts[0] == "layers" && parts[2] == "mlp" => parts[3] switch
            {
                "gate_proj" => $"layers.{parts[1]}.mlp.gate",
                "up_proj" => $"layers.{parts[1]}.mlp.up",
                "down_proj" => $"layers.{parts[1]}.mlp.down",
                _ => null,
            },
            _ when parts.Length == 5 && parts[0] == "layers" && parts[2] == "self_attn" => parts[3] switch
            {
                "q_proj" => $"layers.{parts[1]}.attention.q",
                "k_proj" => $"layers.{parts[1]}.attention.k",
                "v_proj" => $"layers.{parts[1]}.attention.v",
                "o_proj" => $"layers.{parts[1]}.attention.o",
                _ => null,
            },
            _ => null,
        };
        return canonical is null ? null : $"audio.{canonical}.{kind}";
    }

    private static JsonElement AudioConfig(HfConfig config)
        => config.Root.TryGetProperty("audio_config", out var audio) ? audio : throw new NotSupportedException("voxtral_realtime: config.json has no audio_config.");

    private static int Int(JsonElement e, string key, int fallback)
        => e.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : fallback;

    private static (int Rate, int Fft, int Hop, double LogMax) Features(string directory)
    {
        string path = Path.Combine(directory, "processor_config.json");
        if (!File.Exists(path))
        {
            return (16000, 400, 160, 1.5);
        }

        using var document = JsonDocument.Parse(File.ReadAllText(path));
        var root = document.RootElement.TryGetProperty("feature_extractor", out var fe) ? fe : document.RootElement;
        int rate = Int(root, "sampling_rate", 16000);
        double max = root.TryGetProperty("global_log_mel_max", out var m) && m.ValueKind == JsonValueKind.Number ? m.GetDouble() : 1.5;
        return (rate, Int(root, "n_fft", 400), Int(root, "hop_length", 160), max);
    }
}
