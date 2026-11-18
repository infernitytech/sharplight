using System.Text.RegularExpressions;
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
        if (!VibeVoiceHfShared.IsOriginalLayout(c))
        {
            yield return "vibevoice_asr: only the original (Microsoft) checkpoint layout is supported (decoder_config, acoustic_tokenizer_config), not the transformers conversion.";
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
        md[a + "hidden"] = c.Root.GetProperty("decoder_config").GetProperty("hidden_size").GetInt32();
        md[a + "prompt.template"] = Template;
        md[a + "prompt.context_template"] = ContextTemplate;
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
            else if (name.StartsWith("model.acoustic_tokenizer.decoder.", StringComparison.Ordinal))
            {
                // The recognizer never decodes audio.
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

    [GeneratedRegex(@"^model\.(acoustic|semantic)_connector\.(fc1|norm|fc2)\.(weight|bias)$", RegexOptions.CultureInvariant)]
    private static partial Regex ConnectorPattern();
}
