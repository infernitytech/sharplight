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
    /// <summary>The Qwen2 language model of the checkpoint (<c>decoder_config</c> of the original layout, <c>text_config</c> of transformers') as a configuration of its own, with tied embeddings as the checkpoint says.</summary>
    /// <param name="c">The checkpoint's configuration.</param>
    public static HfConfig LanguageModelConfig(HfConfig c)
    {
        var text = LanguageModel(c);
        var node = JsonNode.Parse(text.GetRawText())!.AsObject();
        node["model_type"] = "qwen2";
        node["architectures"] = new JsonArray("Qwen2ForCausalLM");
        if (!node.ContainsKey("tie_word_embeddings") || c.Root.TryGetProperty("tie_word_embeddings", out _) && !c.Root.TryGetProperty("decoder_config", out _))
        {
            node["tie_word_embeddings"] = c.Root.TryGetProperty("tie_word_embeddings", out var tie) && tie.ValueKind == JsonValueKind.True;
        }

        return HfConfig.Parse(node.ToJsonString(), c.Directory);
    }

    /// <summary>The language model's configuration object.</summary>
    /// <param name="c">The checkpoint's configuration.</param>
    public static JsonElement LanguageModel(HfConfig c) => c.Root.TryGetProperty("decoder_config", out var original) ? original : c.Root.GetProperty("text_config");

    /// <summary>The acoustic tokenizer's configuration object (original: <c>acoustic_tokenizer_config</c>; transformers: <c>audio_config</c> of the synthesis model, <c>acoustic_tokenizer_encoder_config</c> of the recognizer).</summary>
    /// <param name="c">The checkpoint's configuration.</param>
    public static JsonElement Acoustic(HfConfig c) => First(c, "acoustic_tokenizer_config", "audio_config", "acoustic_tokenizer_encoder_config");

    /// <summary>The semantic tokenizer's configuration object.</summary>
    /// <param name="c">The checkpoint's configuration.</param>
    public static JsonElement Semantic(HfConfig c) => First(c, "semantic_tokenizer_config", "semantic_model_config", "semantic_tokenizer_encoder_config");

    private static JsonElement First(HfConfig c, params string[] names)
    {
        foreach (string name in names)
        {
            if (c.Root.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.Object)
            {
                return value;
            }
        }

        throw new InvalidDataException($"vibevoice: the configuration has none of {string.Join(", ", names)}.");
    }

    /// <summary>Whether a configuration is one of the layouts these mappings read: the original (Microsoft) one (<c>decoder_config</c>, <c>acoustic_tokenizer_config</c>) or transformers' conversion (<c>text_config</c> and the tokenizers' encoder configurations).</summary>
    /// <param name="c">The checkpoint's configuration.</param>
    public static bool IsSupportedLayout(HfConfig c)
        => c.Root.TryGetProperty("decoder_config", out _) && c.Root.TryGetProperty("acoustic_tokenizer_config", out _)
           || c.Root.TryGetProperty("text_config", out _) && (c.Root.TryGetProperty("audio_config", out _) || c.Root.TryGetProperty("acoustic_tokenizer_encoder_config", out _));

    /// <summary>Adds the hyper-parameters of the tokenizer configuration <paramref name="section"/> as <c>{prefix}.{filters,kernel,depths,ratios,latent,eps,vae_std}</c>.</summary>
    /// <param name="md">Receives the keys.</param>
    /// <param name="prefix">The key prefix, with its trailing dot (<c>vibevoice.acoustic.</c>).</param>
    /// <param name="section">The <c>acoustic_tokenizer_config</c> or <c>semantic_tokenizer_config</c> object.</param>
    public static void AddTokenizer(Dictionary<string, object> md, string prefix, JsonElement section)
    {
        if (section.TryGetProperty("encoder_n_filters", out var original))
        {
            // The original layout: the ratios in the decoder's order, the noise scale of the encoder's sample as fix_std / 0.8.
            md[prefix + "filters"] = original.GetInt32();
            md[prefix + "kernel"] = 7;
            md[prefix + "depths"] = ParseDepths(section.GetProperty("encoder_depths"));
            md[prefix + "ratios"] = section.GetProperty("encoder_ratios").EnumerateArray().Select(v => v.GetInt64()).ToArray();
            md[prefix + "latent"] = section.GetProperty("vae_dim").GetInt32();
            md[prefix + "eps"] = section.TryGetProperty("layernorm_eps", out var eps) ? (float)eps.GetDouble() : 1e-5f;
            md[prefix + "vae_std"] = section.TryGetProperty("fix_std", out var std) && std.ValueKind == JsonValueKind.Number ? (float)(std.GetDouble() / 0.8) : 0f;
            return;
        }

        // Transformers' layout: the downsampling ratios of the encoder (the decoder's in reverse).
        md[prefix + "filters"] = section.GetProperty("num_filters").GetInt32();
        md[prefix + "kernel"] = section.TryGetProperty("kernel_size", out var kernel) ? kernel.GetInt32() : 7;
        md[prefix + "depths"] = ParseDepths(section.GetProperty("depths"));
        md[prefix + "ratios"] = section.GetProperty("downsampling_ratios").EnumerateArray().Select(v => v.GetInt64()).Reverse().ToArray();
        md[prefix + "latent"] = section.GetProperty("hidden_size").GetInt32();
        md[prefix + "eps"] = section.TryGetProperty("rms_norm_eps", out var norm) ? (float)norm.GetDouble() : 1e-5f;
        md[prefix + "vae_std"] = section.TryGetProperty("vae_std", out var vae) ? (float)vae.GetDouble() : 0f;
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

        path = Path.Combine(c.Directory, "processor_config.json");
        if (File.Exists(path))
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(path));
            if (doc.RootElement.TryGetProperty("feature_extractor", out var extractor) && extractor.TryGetProperty("sampling_rate", out var rate))
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
    /// The canonical name of a tensor of either layout that is not the language model's (<c>model.language_model.*</c>, <c>language_model.*</c> and <c>lm_head.weight</c> belong to the language model), or
    /// <see langword="null"/> when the name is not known: the speech tokenizers (<c>{acoustic,semantic}.{encoder,decoder}.…</c>: the convolution modules' wrapper names are dropped,
    /// <c>downsample_layers.i.0</c> / <c>stem</c> / <c>conv_layers.i</c> become <c>down.i</c> / <c>up.i</c>, stages <c>stage.i.j</c>, <c>ffn.linear{1,2}</c> become <c>ffn{1,2}</c>), the connectors
    /// (<c>{acoustic,semantic}_connector.{fc1,norm,fc2}</c>), the diffusion head (<c>diffusion.…</c>) and the latent scale and bias (<c>vibevoice.latent_{scale,bias}</c>).
    /// </summary>
    /// <param name="name">The tensor's name.</param>
    public static string? CanonicalName(string name)
    {
        string n = name.StartsWith("model.", StringComparison.Ordinal) ? name["model.".Length..] : name;
        if (n == "speech_scaling_factor" || n == "latent_scaling_factor")
        {
            return "vibevoice.latent_scale";
        }

        if (n == "speech_bias_factor" || n == "latent_bias_factor")
        {
            return "vibevoice.latent_bias";
        }

        if (OriginalTokenizer().Match(n) is { Success: true } original)
        {
            return OriginalTokenizerName(original.Groups[2].Value) is { } tail ? original.Groups[1].Value + "." + tail : null;
        }

        if (TransformersTokenizer().Match(n) is { Success: true } transformers)
        {
            string kind = transformers.Groups[1].Value == "semantic_tokenizer_encoder" ? "semantic" : "acoustic";
            string side = transformers.Groups[2].Success ? transformers.Groups[2].Value : "encoder";
            return TransformersTokenizerName(transformers.Groups[3].Value, side) is { } tail ? $"{kind}.{tail}" : null;
        }

        if (Connector().Match(n) is { Success: true } connector)
        {
            string role = connector.Groups[1].Success ? connector.Groups[1].Value : connector.Groups[2].Success ? connector.Groups[2].Value : "acoustic";
            string layer = connector.Groups[3].Value;
            return $"{role}_connector.{ConnectorLayer(layer)}.{connector.Groups[4].Value}";
        }

        if (n.StartsWith("prediction_head.", StringComparison.Ordinal) || n.StartsWith("diffusion_head.", StringComparison.Ordinal))
        {
            return "diffusion." + Head(n[(n.IndexOf('.', StringComparison.Ordinal) + 1)..]);
        }

        return null;
    }

    /// <summary>
    /// The name of a language model tensor of either layout as the Qwen2 mapping reads it (<c>model.*</c>, <c>lm_head.weight</c>), or <see langword="null"/> for any other tensor:
    /// <c>model.language_model.*</c> (original; transformers' synthesis model), <c>language_model.model.*</c> and <c>language_model.lm_head.weight</c> (transformers' recognizer, whose language model is
    /// a causal LM of its own), and <c>lm_head.weight</c>.
    /// </summary>
    /// <param name="name">The tensor's name.</param>
    public static string? LanguageName(string name)
    {
        string n = name.StartsWith("model.", StringComparison.Ordinal) ? name["model.".Length..] : name;
        if (n == "lm_head.weight")
        {
            return "lm_head.weight";
        }

        if (!n.StartsWith("language_model.", StringComparison.Ordinal))
        {
            return null;
        }

        string rest = n["language_model.".Length..];
        return rest == "lm_head.weight" ? "lm_head.weight" : rest.StartsWith("model.", StringComparison.Ordinal) ? rest : "model." + rest;
    }

    /// <summary>Whether a tensor is the acoustic tokenizer's decoder (the recognizer does not use it).</summary>
    /// <param name="name">The tensor's name.</param>
    public static bool IsAcousticDecoder(string name)
        => name.StartsWith("model.acoustic_tokenizer.decoder.", StringComparison.Ordinal) || name.StartsWith("model.audio_tower.decoder.", StringComparison.Ordinal)
           || name.StartsWith("audio_tower.decoder.", StringComparison.Ordinal) || name.StartsWith("acoustic_tokenizer.decoder.", StringComparison.Ordinal);

    /// <summary>The canonical weight of a tensor (<see cref="CanonicalName"/>) as a lazily widened view: convolution kernels (three dimensions) are f32, other tensors keep their type.</summary>
    /// <param name="tensor">The tensor.</param>
    /// <param name="canonical">The canonical name.</param>
    public static WeightInfo Weight(SafetensorsTensor tensor, string canonical)
    {
        var info = tensor.ToWeightInfo();
        return info.Shape.Rank == 3 ? HfTransforms.ToF32(info, canonical) : HfTransforms.Plain(info, canonical);
    }

    private static string? OriginalTokenizerName(string rest)
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
        return m.Success ? $"{m.Groups[1].Value}.stage.{m.Groups[2].Value}.{m.Groups[3].Value}.{BlockTail(m.Groups[4].Value)}" : null;
    }

    // Transformers' layout below the encoder or decoder module: stem.*, conv_layers.i.*, head.*.
    private static string? TransformersTokenizerName(string rest, string side)
    {
        string kind = side == "encoder" ? "down" : "up";
        var m = TransformersConv().Match(rest);
        if (m.Success)
        {
            return m.Groups[1].Value == "stem" ? $"{side}.{kind}.0.{m.Groups[3].Value}" : $"{side}.{kind}.{int.Parse(m.Groups[2].Value, System.Globalization.CultureInfo.InvariantCulture) + 1}.{m.Groups[3].Value}";
        }

        m = TransformersStage().Match(rest);
        if (m.Success)
        {
            int stage = m.Groups[1].Value == "stem" ? 0 : int.Parse(m.Groups[2].Value, System.Globalization.CultureInfo.InvariantCulture) + 1;
            return $"{side}.stage.{stage}.{m.Groups[3].Value}.{BlockTail(m.Groups[4].Value)}";
        }

        m = TransformersHead().Match(rest);
        return m.Success ? $"{side}.head.{m.Groups[1].Value}" : null;
    }

    private static string BlockTail(string tail)
        => tail.Replace("mixer.conv.conv.conv.", "mixer.", StringComparison.Ordinal).Replace("mixer.conv.", "mixer.", StringComparison.Ordinal)
            .Replace("ffn.linear1.", "ffn1.", StringComparison.Ordinal).Replace("ffn.linear2.", "ffn2.", StringComparison.Ordinal);

    private static string ConnectorLayer(string layer) => layer switch
    {
        "fc1" or "linear_1" => "fc1",
        "norm" or "act" => "norm",
        _ => "fc2",
    };

    // The head's tensors: the timestep MLP and the modulation layers get readable names.
    private static string Head(string name)
    {
        name = name.Replace("t_embedder.mlp.0.", "time.fc1.", StringComparison.Ordinal).Replace("t_embedder.mlp.2.", "time.fc2.", StringComparison.Ordinal)
            .Replace("timestep_proj.fc1.", "time.fc1.", StringComparison.Ordinal).Replace("timestep_proj.fc2.", "time.fc2.", StringComparison.Ordinal)
            .Replace("noisy_images_proj.", "noisy_proj.", StringComparison.Ordinal);
        name = Modulation().Replace(name, "layers.$1.ada.");
        name = name.Replace("final_layer.adaLN_modulation.1.", "final.ada.", StringComparison.Ordinal).Replace("final_layer.linear.", "final.out.", StringComparison.Ordinal)
            .Replace("final_layer.linear_1.", "final.ada.", StringComparison.Ordinal).Replace("final_layer.linear_2.", "final.out.", StringComparison.Ordinal);
        return name.Replace("ffn.gate_proj.", "ffn.gate.", StringComparison.Ordinal).Replace("ffn.up_proj.", "ffn.up.", StringComparison.Ordinal)
            .Replace("ffn.down_proj.", "ffn.down.", StringComparison.Ordinal);
    }

    private static int[] ParseDepths(JsonElement element)
        => element.ValueKind == JsonValueKind.String ? element.GetString()!.Split('-').Select(int.Parse).ToArray() : element.EnumerateArray().Select(v => v.GetInt32()).ToArray();

    [GeneratedRegex(@"^(acoustic|semantic)_tokenizer\.(.+)$", RegexOptions.CultureInvariant)]
    private static partial Regex OriginalTokenizer();

    [GeneratedRegex(@"^(audio_tower|acoustic_tokenizer_encoder|semantic_tokenizer_encoder)\.(?:(encoder|decoder)\.)?(.+)$", RegexOptions.CultureInvariant)]
    private static partial Regex TransformersTokenizer();

    [GeneratedRegex(@"^(?:(acoustic|semantic)_connector|multi_modal_projector)\.(?:(acoustic|semantic)_)?(fc1|norm|fc2|linear_1|act|linear_2)\.(weight|bias)$", RegexOptions.CultureInvariant)]
    private static partial Regex Connector();

    [GeneratedRegex(@"^(stem|conv_layers\.(\d+))\.(?:conv\.conv|convtr\.convtr)\.(weight|bias)$", RegexOptions.CultureInvariant)]
    private static partial Regex TransformersConv();

    [GeneratedRegex(@"^(stem|conv_layers\.(\d+))\.stage\.(\d+)\.(.+)$", RegexOptions.CultureInvariant)]
    private static partial Regex TransformersStage();

    [GeneratedRegex(@"^head\.conv\.(weight|bias)$", RegexOptions.CultureInvariant)]
    private static partial Regex TransformersHead();

    [GeneratedRegex(@"^layers\.(\d+)\.(?:adaLN_modulation\.1|linear)\.", RegexOptions.CultureInvariant)]
    private static partial Regex Modulation();

    [GeneratedRegex(@"^(encoder|decoder)\.head\.conv\.conv\.(weight|bias)$", RegexOptions.CultureInvariant)]
    private static partial Regex HeadPattern();

    [GeneratedRegex(@"^(encoder|decoder)\.(downsample_layers|upsample_layers)\.(\d+)\.0\.(?:conv\.conv|convtr\.convtr)\.(weight|bias)$", RegexOptions.CultureInvariant)]
    private static partial Regex LayerPattern();

    [GeneratedRegex(@"^(encoder|decoder)\.stages\.(\d+)\.(\d+)\.(.+)$", RegexOptions.CultureInvariant)]
    private static partial Regex StagePattern();
}
