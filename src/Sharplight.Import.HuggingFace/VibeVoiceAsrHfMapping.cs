using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// VibeVoice speech recognition (<c>VibeVoiceForASRTraining</c> in Microsoft's original layout, <c>model_type</c> <c>vibevoice</c>: a Qwen2 language model, the acoustic and the semantic speech
/// tokenizer and two connectors; the acoustic tokenizer's decoder and any diffusion head are not used and not imported) → canonical <c>vibevoice_asr</c> (documented on
/// <c>VibeVoiceAsrAdapter</c>). Shares the language model and tokenizer mapping with <see cref="VibeVoiceHfMapping"/> (<see cref="VibeVoiceHfShared"/>). The checkpoint ships no tokenizer:
/// the Qwen2.5 <c>tokenizer.json</c> must sit next to it.
/// </summary>
internal sealed partial class VibeVoiceAsrHfMapping : IHfArchitectureMapping
{
    /// <summary>The <c>model_type</c> the importer reports for the ASR class (the config says <c>vibevoice</c>, like the speech synthesis one).</summary>
    internal const string ModelType = "vibevoice_asr";

    private const string Template = "<|im_start|>system\nYou are a helpful assistant that transcribes audio input into text output in JSON format.<|im_end|>\n<|im_start|>user\n"
        + "<|object_ref_start|>{audio}<|object_ref_end|>\nThis is a {duration} seconds audio, please transcribe it with these keys: Start time, End time, Speaker ID, Content<|im_end|>\n";

    private const string ContextTemplate = "<|im_start|>system\nYou are a helpful assistant that transcribes audio input into text output in JSON format.<|im_end|>\n<|im_start|>user\n"
        + "<|object_ref_start|>{audio}<|object_ref_end|>\nThis is a {duration} seconds audio, with extra info: {context}\n\nPlease transcribe it with these keys: Start time, End time, Speaker ID, Content<|im_end|>\n";

    /// <inheritdoc/>
    public string Architecture => "vibevoice_asr";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = [ModelType];

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig c)
    {
        if (!VibeVoiceHfShared.IsSupportedLayout(c))
        {
            yield return "vibevoice_asr: the configuration is neither the original (Microsoft) layout (decoder_config, acoustic_tokenizer_config) nor transformers' conversion (text_config, acoustic_tokenizer_encoder_config).";
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
        md[a + "audio_token_id"] = VibeVoiceHfShared.TokenId(c, "<|box_start|>", 151648);
        md[a + "audio_bos_token_id"] = VibeVoiceHfShared.TokenId(c, "<|object_ref_start|>", 151646);
        md[a + "audio_eos_token_id"] = VibeVoiceHfShared.TokenId(c, "<|object_ref_end|>", 151647);
        md[a + "im_end_token_id"] = VibeVoiceHfShared.TokenId(c, "<|im_end|>", 151645);
        md[a + "eos_token_id"] = VibeVoiceHfShared.TokenId(c, "<|endoftext|>", 151643);
        md[a + "sample_rate"] = VibeVoiceHfShared.SampleRate(c);
        md[a + "normalize_dbfs"] = -25f;
        md[a + "hidden"] = VibeVoiceHfShared.LanguageModel(c).GetProperty("hidden_size").GetInt32();
        md[a + "prompt.template"] = Template;
        md[a + "prompt.context_template"] = ContextTemplate;
        VibeVoiceHfShared.AddTokenizer(md, a + "acoustic.", VibeVoiceHfShared.Acoustic(c));
        VibeVoiceHfShared.AddTokenizer(md, a + "semantic.", VibeVoiceHfShared.Semantic(c));
        return md;
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
            else if (VibeVoiceHfShared.IsAcousticDecoder(name) || name.Contains("diffusion_head.", StringComparison.Ordinal) || name.Contains("prediction_head.", StringComparison.Ordinal))
            {
                // The recognizer never decodes audio and has no use for a diffusion head.
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
