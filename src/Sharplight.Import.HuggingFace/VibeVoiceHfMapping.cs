using System.Text.Json;
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
        if (!VibeVoiceHfShared.IsSupportedLayout(c) || !c.Root.TryGetProperty("diffusion_head_config", out var head))
        {
            yield return "vibevoice: the configuration is neither the original (Microsoft) layout (decoder_config, acoustic_tokenizer_config, diffusion_head_config) nor transformers' conversion (text_config, audio_config, diffusion_head_config).";
            yield break;
        }

        foreach (string reason in VibeVoiceHfShared.ValidateTokenizer("acoustic tokenizer", VibeVoiceHfShared.Acoustic(c)))
        {
            yield return reason;
        }

        foreach (string reason in VibeVoiceHfShared.ValidateTokenizer("semantic tokenizer", VibeVoiceHfShared.Semantic(c)))
        {
            yield return reason;
        }

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
        var text = VibeVoiceHfShared.LanguageModel(c);
        var head = c.Root.GetProperty("diffusion_head_config");
        var generation = GenerationSettings(c);
        md[a + "audio_bos_token_id"] = VibeVoiceHfShared.TokenId(c, "<|vision_start|>", 151652);
        md[a + "audio_eos_token_id"] = VibeVoiceHfShared.TokenId(c, "<|vision_end|>", 151653);
        md[a + "audio_token_id"] = VibeVoiceHfShared.TokenId(c, "<|vision_pad|>", 151654);
        md[a + "eos_token_id"] = VibeVoiceHfShared.TokenId(c, "<|endoftext|>", 151643);
        md[a + "sample_rate"] = VibeVoiceHfShared.SampleRate(c);
        md[a + "normalize_dbfs"] = -25f;
        md[a + "hidden"] = text.GetProperty("hidden_size").GetInt32();
        md[a + "diffusion.layers"] = head.TryGetProperty("head_layers", out var layers) ? layers.GetInt32() : head.GetProperty("num_hidden_layers").GetInt32();
        md[a + "diffusion.frequencies"] = head.TryGetProperty("frequency_embedding_size", out var frequencies) ? frequencies.GetInt32() : 256;
        md[a + "diffusion.eps"] = head.TryGetProperty("rms_norm_eps", out var eps) ? (float)eps.GetDouble() : 1e-5f;
        md[a + "diffusion.train_steps"] = head.TryGetProperty("ddpm_num_steps", out var steps) ? steps.GetInt32() : 1000;
        md[a + "diffusion.max_period"] = head.TryGetProperty("diffusion_max_period", out var period) ? (float)period.GetDouble() : 10000f;
        md[a + "diffusion.beta_schedule"] = head.TryGetProperty("ddpm_beta_schedule", out var schedule) && schedule.GetString() is { } named ? (named == "cosine" ? "squaredcos_cap_v2" : named) : generation.BetaSchedule;
        md[a + "diffusion.prediction_type"] = head.TryGetProperty("prediction_type", out var prediction) ? prediction.GetString()! : generation.PredictionType;
        md[a + "sampling.cfg_scale"] = generation.Guidance;
        md[a + "sampling.diffusion_steps"] = generation.Steps;
        md[a + "sampling.max_frames"] = generation.MaxFrames;
        md[a + "prompt.system"] = SystemPrompt;
        md[a + "prompt.voice"] = VoicePrompt;
        md[a + "prompt.text"] = TextPrompt;
        md[a + "prompt.speech"] = SpeechPrompt;
        VibeVoiceHfShared.AddTokenizer(md, a + "acoustic.", VibeVoiceHfShared.Acoustic(c));
        VibeVoiceHfShared.AddTokenizer(md, a + "semantic.", VibeVoiceHfShared.Semantic(c));
        return md;
    }

    // The sampling defaults of transformers' <c>generation_config.json</c> (guidance scale, diffusion steps, noise scheduler); the original checkpoint has none (the demo's 1.3 and 10).
    private static (float Guidance, float Steps, float MaxFrames, string BetaSchedule, string PredictionType) GenerationSettings(HfConfig c)
    {
        float guidance = 1.3f;
        float steps = 10f;
        float maxFrames = 40500f;
        string schedule = "squaredcos_cap_v2";
        string prediction = "v_prediction";
        string path = Path.Combine(c.Directory, "generation_config.json");
        if (File.Exists(path))
        {
            using var document = JsonDocument.Parse(File.ReadAllText(path));
            var root = document.RootElement;
            guidance = root.TryGetProperty("guidance_scale", out var g) && g.ValueKind == JsonValueKind.Number ? (float)g.GetDouble() : guidance;
            steps = root.TryGetProperty("num_diffusion_steps", out var n) && n.ValueKind == JsonValueKind.Number ? (float)n.GetDouble() : steps;
            maxFrames = root.TryGetProperty("max_new_tokens", out var m) && m.ValueKind == JsonValueKind.Number ? (float)m.GetDouble() : maxFrames;
            if (root.TryGetProperty("noise_scheduler_config", out var scheduler))
            {
                schedule = scheduler.TryGetProperty("beta_schedule", out var b) ? b.GetString() ?? schedule : schedule;
                prediction = scheduler.TryGetProperty("prediction_type", out var t) ? t.GetString() ?? prediction : prediction;
            }
        }

        return (guidance, steps, maxFrames, schedule, prediction);
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        var language = new Dictionary<string, SafetensorsTensor>(StringComparer.Ordinal);
        var collector = new HfWeightCollector();
        foreach (var (name, tensor) in tensors)
        {
            if (VibeVoiceHfShared.LanguageName(name) is { } languageName)
            {
                language[languageName] = tensor;
            }
            else if (VibeVoiceHfShared.CanonicalName(name) is { } canonical)
            {
                collector.Add(VibeVoiceHfShared.Weight(tensor, canonical));
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
}
