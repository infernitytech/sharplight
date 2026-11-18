using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// VibeVoice text to speech (<c>vibevoice</c>, Microsoft's original layout of <c>VibeVoiceForConditionalGeneration</c>: a Qwen2 language model, a diffusion prediction head, the acoustic
/// tokenizer's encoder and decoder, the semantic tokenizer's encoder and the two connectors) → canonical <c>vibevoice</c> (documented on <c>VibeVoiceAdapter</c>). The language model maps
/// through <see cref="DecoderHfMapping"/> as a Qwen2 (its tied embedding stays tied; the host reads token rows from it), the tokenizers through <see cref="VibeVoiceHfShared"/>, the head
/// and the connectors by renaming. The checkpoint ships no tokenizer: the Qwen2.5 <c>tokenizer.json</c> (with the audio tokens <c>&lt;|vision_start|&gt;</c>, <c>&lt;|vision_end|&gt;</c>,
/// <c>&lt;|vision_pad|&gt;</c>) must sit next to it.
/// </summary>
internal sealed partial class VibeVoiceHfMapping : IHfArchitectureMapping
{
    private const string SystemPrompt = " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n";
    private const string VoicePrompt = " Voice input:\n";
    private const string TextPrompt = " Text input:\n";
    private const string SpeechPrompt = " Speech output:\n";

    /// <inheritdoc/>
    public string Architecture => "vibevoice";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["vibevoice"];

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig c)
    {
        if (!VibeVoiceHfShared.IsOriginalLayout(c))
        {
            yield return "vibevoice: only the original (Microsoft) checkpoint layout is supported (decoder_config, acoustic_tokenizer_config), not the transformers conversion.";
            yield break;
        }

        foreach (string reason in VibeVoiceHfShared.ValidateTokenizer("acoustic tokenizer", c.Root.GetProperty("acoustic_tokenizer_config")))
        {
            yield return reason;
        }

        foreach (string reason in VibeVoiceHfShared.ValidateTokenizer("semantic tokenizer", c.Root.GetProperty("semantic_tokenizer_config")))
        {
            yield return reason;
        }

        var head = c.Root.GetProperty("diffusion_head_config");
        if (head.TryGetProperty("diffusion_type", out var type) && type.GetString() != "ddpm")
        {
            yield return $"vibevoice: only the DDPM diffusion head is supported, not '{type.GetString()}'.";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        var md = new Dictionary<string, object>(StringComparer.Ordinal);
        using (var config = VibeVoiceHfShared.LanguageModelConfig(c))
        {
            foreach (var (key, value) in new DecoderHfMapping().MapHyperParameters(config, options))
            {
                md[key] = value;
            }
        }

        string a = Architecture + ".";
        var text = c.Root.GetProperty("decoder_config");
        var head = c.Root.GetProperty("diffusion_head_config");
        md[a + "audio_bos_token_id"] = VibeVoiceHfShared.TokenId(c, "<|vision_start|>", 151652);
        md[a + "audio_eos_token_id"] = VibeVoiceHfShared.TokenId(c, "<|vision_end|>", 151653);
        md[a + "audio_token_id"] = VibeVoiceHfShared.TokenId(c, "<|vision_pad|>", 151654);
        md[a + "eos_token_id"] = VibeVoiceHfShared.TokenId(c, "<|endoftext|>", 151643);
        md[a + "sample_rate"] = VibeVoiceHfShared.SampleRate(c);
        md[a + "normalize_dbfs"] = -25f;
        md[a + "hidden"] = text.GetProperty("hidden_size").GetInt32();
        md[a + "diffusion.layers"] = head.GetProperty("head_layers").GetInt32();
        md[a + "diffusion.frequencies"] = 256;
        md[a + "diffusion.eps"] = head.TryGetProperty("rms_norm_eps", out var eps) ? (float)eps.GetDouble() : 1e-5f;
        md[a + "diffusion.train_steps"] = head.GetProperty("ddpm_num_steps").GetInt32();
        md[a + "diffusion.max_period"] = 10000f;
        md[a + "diffusion.beta_schedule"] = head.GetProperty("ddpm_beta_schedule").GetString()!;
        md[a + "diffusion.prediction_type"] = head.GetProperty("prediction_type").GetString()!;
        md[a + "sampling.cfg_scale"] = 1.3f;
        md[a + "sampling.diffusion_steps"] = 10f;
        md[a + "sampling.max_frames"] = 40500f;
        md[a + "prompt.system"] = SystemPrompt;
        md[a + "prompt.voice"] = VoicePrompt;
        md[a + "prompt.text"] = TextPrompt;
        md[a + "prompt.speech"] = SpeechPrompt;
        VibeVoiceHfShared.AddTokenizer(md, a + "acoustic.", c.Root.GetProperty("acoustic_tokenizer_config"));
        VibeVoiceHfShared.AddTokenizer(md, a + "semantic.", c.Root.GetProperty("semantic_tokenizer_config"));
        return md;
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        var language = new Dictionary<string, SafetensorsTensor>(StringComparer.Ordinal);
        var collector = new HfWeightCollector();
        foreach (var (name, tensor) in tensors)
        {
            if (name.StartsWith("model.language_model.", StringComparison.Ordinal))
            {
                language["model." + name["model.language_model.".Length..]] = tensor;
            }
            else if (name == "lm_head.weight")
            {
                language[name] = tensor;
            }
            else if (name.StartsWith("model.acoustic_tokenizer.", StringComparison.Ordinal) || name.StartsWith("model.semantic_tokenizer.", StringComparison.Ordinal))
            {
                bool acoustic = name.StartsWith("model.acoustic_tokenizer.", StringComparison.Ordinal);
                string rest = name[(acoustic ? "model.acoustic_tokenizer.".Length : "model.semantic_tokenizer.".Length)..];
                var weight = VibeVoiceHfShared.Tokenizer(tensor, rest, acoustic ? "acoustic." : "semantic.");
                if (weight is null)
                {
                    collector.AddUnmapped(name);
                }
                else
                {
                    collector.Add(weight);
                }
            }
            else if (ConnectorPattern().Match(name) is { Success: true } connector)
            {
                collector.Add(HfTransforms.Plain(tensor.ToWeightInfo(), connector.Groups[1].Value + "_connector." + connector.Groups[2].Value + "." + connector.Groups[3].Value));
            }
            else if (name.StartsWith("model.prediction_head.", StringComparison.Ordinal))
            {
                collector.Add(HfTransforms.Plain(tensor.ToWeightInfo(), "diffusion." + Head(name["model.prediction_head.".Length..])));
            }
            else if (name == "model.speech_scaling_factor")
            {
                collector.Add(HfTransforms.Plain(tensor.ToWeightInfo(), "vibevoice.latent_scale"));
            }
            else if (name == "model.speech_bias_factor")
            {
                collector.Add(HfTransforms.Plain(tensor.ToWeightInfo(), "vibevoice.latent_bias"));
            }
            else
            {
                collector.AddUnmapped(name);
            }
        }

        using (var config = VibeVoiceHfShared.LanguageModelConfig(c))
        {
            foreach (var (_, w) in new DecoderHfMapping().MapWeights(config, language, options))
            {
                collector.Add(w);
            }
        }

        collector.ThrowIfUnmapped(c.ModelType);
        return collector.Weights;
    }

    // The head's tensors: the timestep MLP and the modulation layers get readable names.
    private static string Head(string name)
    {
        name = name.Replace("t_embedder.mlp.0.", "time.fc1.", StringComparison.Ordinal).Replace("t_embedder.mlp.2.", "time.fc2.", StringComparison.Ordinal)
            .Replace("noisy_images_proj.", "noisy_proj.", StringComparison.Ordinal);
        name = Layer().Replace(name, "layers.$1.ada.");
        name = name.Replace("final_layer.adaLN_modulation.1.", "final.ada.", StringComparison.Ordinal).Replace("final_layer.linear.", "final.out.", StringComparison.Ordinal);
        return name.Replace("ffn.gate_proj.", "ffn.gate.", StringComparison.Ordinal).Replace("ffn.up_proj.", "ffn.up.", StringComparison.Ordinal)
            .Replace("ffn.down_proj.", "ffn.down.", StringComparison.Ordinal);
    }

    [GeneratedRegex(@"^model\.(acoustic|semantic)_connector\.(fc1|norm|fc2)\.(weight|bias)$", RegexOptions.CultureInvariant)]
    private static partial Regex ConnectorPattern();

    [GeneratedRegex(@"^layers\.(\d+)\.adaLN_modulation\.1\.", RegexOptions.CultureInvariant)]
    private static partial Regex Layer();
}
