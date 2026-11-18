using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// What the two VibeVoice checkpoints (<c>VibeVoiceForConditionalGeneration</c>: text to speech, <c>VibeVoiceForASRTraining</c>: speech to text) share: the Qwen2 language model's view
/// of the configuration, the hyper-parameters of the acoustic and semantic speech tokenizers, and the tokenizers' and connectors' weight names.
/// </summary>
internal static partial class VibeVoiceHfShared
{
    /// <summary>The Qwen2 language model of the checkpoint (<c>decoder_config</c>) as a configuration of its own, with tied embeddings as the checkpoint says.</summary>
    /// <param name="c">The checkpoint's configuration.</param>
    public static HfConfig LanguageModelConfig(HfConfig c)
    {
        var node = JsonNode.Parse(c.Root.GetProperty("decoder_config").GetRawText())!.AsObject();
        node["model_type"] = "qwen2";
        node["architectures"] = new JsonArray("Qwen2ForCausalLM");
        node.TryAdd("tie_word_embeddings", false);
        return HfConfig.Parse(node.ToJsonString(), c.Directory);
    }

    /// <summary>Whether a configuration is the original (Microsoft) layout this mapping reads: a <c>decoder_config</c> next to the tokenizers' configurations.</summary>
    /// <param name="c">The checkpoint's configuration.</param>
    public static bool IsOriginalLayout(HfConfig c) => c.Root.TryGetProperty("decoder_config", out _) && c.Root.TryGetProperty("acoustic_tokenizer_config", out _);

    /// <summary>Adds the hyper-parameters of the tokenizer configuration <paramref name="section"/> as <c>{prefix}.{filters,kernel,depths,ratios,latent,eps,vae_std}</c>.</summary>
    /// <param name="md">Receives the keys.</param>
    /// <param name="prefix">The key prefix, with its trailing dot (<c>vibevoice.acoustic.</c>).</param>
    /// <param name="section">The <c>acoustic_tokenizer_config</c> or <c>semantic_tokenizer_config</c> object.</param>
    public static void AddTokenizer(Dictionary<string, object> md, string prefix, JsonElement section)
    {
        md[prefix + "filters"] = section.GetProperty("encoder_n_filters").GetInt32();
        md[prefix + "kernel"] = 7;
        md[prefix + "depths"] = ParseDepths(section.GetProperty("encoder_depths"));
        md[prefix + "ratios"] = section.GetProperty("encoder_ratios").EnumerateArray().Select(v => v.GetInt64()).ToArray();
        md[prefix + "latent"] = section.GetProperty("vae_dim").GetInt32();
        md[prefix + "eps"] = section.TryGetProperty("layernorm_eps", out var eps) ? (float)eps.GetDouble() : 1e-5f;
        md[prefix + "vae_std"] = section.TryGetProperty("fix_std", out var std) && std.ValueKind == JsonValueKind.Number ? (float)(std.GetDouble() / 0.8) : 0f;
    }

    /// <summary>Whether the tokenizer configuration is one this mapping runs: causal convolutions, zero padding, RMS norm, depthwise-convolution mixers, no convolution normalization.</summary>
    /// <param name="name">The configuration's name for the message.</param>
    /// <param name="section">The configuration.</param>
    public static IEnumerable<string> ValidateTokenizer(string name, JsonElement section)
    {
        if (section.TryGetProperty("causal", out var causal) && causal.ValueKind == JsonValueKind.False)
        {
            yield return $"vibevoice: the {name} must be causal.";
        }

        if (section.TryGetProperty("mixer_layer", out var mixer) && mixer.GetString() != "depthwise_conv")
        {
            yield return $"vibevoice: only depthwise-convolution mixers are supported, not '{mixer.GetString()}'.";
        }

        if (section.TryGetProperty("layernorm", out var norm) && norm.GetString() != "RMSNorm")
        {
            yield return $"vibevoice: only RMS norm is supported in the {name}, not '{norm.GetString()}'.";
        }

        if (section.TryGetProperty("conv_norm", out var convNorm) && convNorm.GetString() is { } value && value != "none")
        {
            yield return $"vibevoice: convolution normalization '{value}' is not supported.";
        }

        if (section.TryGetProperty("pad_mode", out var pad) && pad.GetString() != "constant")
        {
            yield return $"vibevoice: only constant padding is supported in the {name}.";
        }
    }

    /// <summary>The sample rate of the audio processor (<c>preprocessor_config.json</c>), 24000 when it says nothing.</summary>
    /// <param name="c">The checkpoint's configuration.</param>
    public static int SampleRate(HfConfig c)
    {
        string path = Path.Combine(c.Directory, "preprocessor_config.json");
        if (File.Exists(path))
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(path));
            if (doc.RootElement.TryGetProperty("audio_processor", out var audio) && audio.TryGetProperty("sampling_rate", out var rate))
            {
                return rate.GetInt32();
            }
        }

        return 24000;
    }

    /// <summary>The id of a special token of the checkpoint's <c>tokenizer.json</c>, or <paramref name="fallback"/> when there is none.</summary>
    /// <param name="c">The checkpoint's configuration.</param>
    /// <param name="token">The token's text.</param>
    /// <param name="fallback">The id Qwen2.5's vocabulary gives it.</param>
    public static int TokenId(HfConfig c, string token, int fallback)
    {
        string path = Path.Combine(c.Directory, "tokenizer.json");
        if (File.Exists(path))
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(path));
            if (doc.RootElement.TryGetProperty("added_tokens", out var added))
            {
                foreach (var entry in added.EnumerateArray())
                {
                    if (entry.GetProperty("content").GetString() == token)
                    {
                        return entry.GetProperty("id").GetInt32();
                    }
                }
            }
        }

        return fallback;
    }

    /// <summary>
    /// Maps the weight of a speech tokenizer (original names below <c>model.{acoustic,semantic}_tokenizer.</c>) to <c>{canonical}.{encoder,decoder}.…</c>: the convolution modules' wrapper
    /// names are dropped, <c>downsample_layers.i.0</c> / <c>upsample_layers.i.0</c> become <c>down.i</c> / <c>up.i</c>, <c>stages.i.j</c> become <c>stage.i.j</c>, <c>ffn.linear{1,2}</c> become
    /// <c>ffn{1,2}</c>; convolution kernels (three dimensions) are widened to f32.
    /// </summary>
    /// <param name="tensor">The tensor.</param>
    /// <param name="rest">The name below the tokenizer's prefix.</param>
    /// <param name="canonical">The canonical prefix (<c>acoustic.</c> or <c>semantic.</c>).</param>
    public static WeightInfo? Tokenizer(SafetensorsTensor tensor, string rest, string canonical)
    {
        string? name = TokenizerName(rest);
        if (name is null)
        {
            return null;
        }

        var info = tensor.ToWeightInfo();
        return info.Shape.Rank == 3 ? HfTransforms.ToF32(info, canonical + name) : HfTransforms.Plain(info, canonical + name);
    }

    private static string? TokenizerName(string rest)
    {
        var m = HeadPattern().Match(rest);
        if (m.Success)
        {
            return $"{m.Groups[1].Value}.head.{m.Groups[2].Value}";
        }

        m = LayerPattern().Match(rest);
        if (m.Success)
        {
            string kind = m.Groups[2].Value == "downsample_layers" ? "down" : "up";
            return $"{m.Groups[1].Value}.{kind}.{m.Groups[3].Value}.{m.Groups[4].Value}";
        }

        m = StagePattern().Match(rest);
        if (m.Success)
        {
            string tail = m.Groups[4].Value.Replace("mixer.conv.conv.conv.", "mixer.", StringComparison.Ordinal).Replace("ffn.linear1.", "ffn1.", StringComparison.Ordinal)
                .Replace("ffn.linear2.", "ffn2.", StringComparison.Ordinal);
            return $"{m.Groups[1].Value}.stage.{m.Groups[2].Value}.{m.Groups[3].Value}.{tail}";
        }

        return null;
    }

    private static int[] ParseDepths(JsonElement element)
        => element.ValueKind == JsonValueKind.String ? element.GetString()!.Split('-').Select(int.Parse).ToArray() : element.EnumerateArray().Select(v => v.GetInt32()).ToArray();

    [GeneratedRegex(@"^(encoder|decoder)\.head\.conv\.conv\.(weight|bias)$", RegexOptions.CultureInvariant)]
    private static partial Regex HeadPattern();

    [GeneratedRegex(@"^(encoder|decoder)\.(downsample_layers|upsample_layers)\.(\d+)\.0\.(?:conv\.conv|convtr\.convtr)\.(weight|bias)$", RegexOptions.CultureInvariant)]
    private static partial Regex LayerPattern();

    [GeneratedRegex(@"^(encoder|decoder)\.stages\.(\d+)\.(\d+)\.(.+)$", RegexOptions.CultureInvariant)]
    private static partial Regex StagePattern();
}
