using System.Text.Json;
using System.Text.Json.Nodes;
using Sharplight.Models;
using Sharplight.Tensors;
using Sharplight.Vision;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Qwen3-TTS (<c>qwen3_tts</c>, Qwen3TTSForConditionalGeneration) → canonical <c>qwen3_tts</c> (documented on <c>Qwen3TtsAdapter</c>): the talker (a Qwen3 decoder over
/// codec tokens, fed with embedding rows) maps through <see cref="DecoderHfMapping"/> with its codec embedding as the vocabulary and <c>codec_head</c> as the head; the code
/// predictor (a small Qwen3 decoder that predicts codebooks 1 to 15 of a frame) the same way with its hyper-parameters under <c>qwen3_tts.code_predictor.</c> and its weights
/// under <c>code_predictor.</c>; the text embedding, the text projection, the code predictor's codebook embeddings and heads (read on the host) keep their names under
/// <c>qwen3_tts.</c>. The speech tokenizer in <c>speech_tokenizer/</c> is the model's vocoder companion (<see cref="Qwen3TtsTokenizerHfMapping"/>).
/// </summary>
internal sealed class Qwen3TtsHfMapping : IHfArchitectureMapping
{
    /// <summary>The folder of the speech tokenizer (the 12 Hz codec) inside the model folder.</summary>
    public const string SpeechTokenizerFolder = "speech_tokenizer";

    private const string Talker = "talker.";
    private const string TalkerModel = "talker.model.";
    private const string CodePredictor = "talker.code_predictor.";

    /// <inheritdoc/>
    public string Architecture => "qwen3_tts";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["qwen3_tts"];

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig c)
    {
        string type = c.Root.TryGetProperty("tts_model_type", out var t) ? t.GetString() ?? "" : "";
        if (type != "custom_voice")
        {
            yield return $"qwen3_tts: only the CustomVoice models (predefined speakers) are supported, not '{type}' (voice cloning and voice design are not implemented).";
        }

        if (c.Root.TryGetProperty("tokenizer_type", out var tokenizer) && tokenizer.GetString() != "qwen3_tts_tokenizer_12hz")
        {
            yield return $"qwen3_tts: only the 12 Hz speech tokenizer is supported, not '{tokenizer.GetString()}'.";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        var talker = c.Root.GetProperty("talker_config");
        var predictor = talker.GetProperty("code_predictor_config");
        var md = new Dictionary<string, object>(StringComparer.Ordinal);
        var mapping = new DecoderHfMapping();
        using (var config = Qwen3Config(talker, c.Directory))
        {
            foreach (var (key, value) in mapping.MapHyperParameters(config, options))
            {
                md[key] = value;
            }
        }

        using (var config = Qwen3Config(predictor, c.Directory))
        {
            foreach (var (key, value) in mapping.MapHyperParameters(config, options))
            {
                md[Architecture + ".code_predictor." + key] = value;
            }
        }

        string a = Architecture + ".";
        md[a + "text_bos"] = c.Int("tts_bos_token_id");
        md[a + "text_eos"] = c.Int("tts_eos_token_id");
        md[a + "text_pad"] = c.Int("tts_pad_token_id");
        md[a + "text_hidden_size"] = talker.GetProperty("text_hidden_size").GetInt32();
        md[a + "code_groups"] = talker.GetProperty("num_code_groups").GetInt32();
        md[a + "codec_bos"] = talker.GetProperty("codec_bos_id").GetInt32();
        md[a + "codec_eos"] = talker.GetProperty("codec_eos_token_id").GetInt32();
        md[a + "codec_pad"] = talker.GetProperty("codec_pad_id").GetInt32();
        md[a + "codec_think"] = talker.GetProperty("codec_think_id").GetInt32();
        md[a + "codec_nothink"] = talker.GetProperty("codec_nothink_id").GetInt32();
        md[a + "codec_think_bos"] = talker.GetProperty("codec_think_bos_id").GetInt32();
        md[a + "codec_think_eos"] = talker.GetProperty("codec_think_eos_id").GetInt32();
        var languages = talker.GetProperty("codec_language_id").EnumerateObject().ToList();
        md[a + "languages"] = languages.Select(p => p.Name).ToArray();
        md[a + "language_ids"] = languages.Select(p => p.Value.GetInt32()).ToArray();
        var speakers = talker.GetProperty("spk_id").EnumerateObject().ToList();
        var dialects = talker.TryGetProperty("spk_is_dialect", out var d) ? d : default;
        md[a + "speakers"] = speakers.Select(p => p.Name).ToArray();
        md[a + "speaker_ids"] = speakers.Select(p => p.Value.GetInt32()).ToArray();
        md[a + "speaker_dialects"] = speakers.Select(p => dialects.ValueKind == JsonValueKind.Object && dialects.TryGetProperty(p.Name, out var v) && v.ValueKind == JsonValueKind.String
            ? v.GetString()!
            : "").ToArray();
        string generation = Path.Combine(c.Directory, "generation_config.json");
        if (File.Exists(generation))
        {
            using var g = JsonDocument.Parse(File.ReadAllText(generation));
            foreach (var (key, name) in new[] { ("temperature", "temperature"), ("top_k", "top_k"), ("top_p", "top_p"), ("repetition_penalty", "repetition_penalty"),
                ("subtalker_temperature", "code_predictor.temperature"), ("subtalker_top_k", "code_predictor.top_k"), ("subtalker_top_p", "code_predictor.top_p") })
            {
                if (g.RootElement.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.Number)
                {
                    md[a + "sampling." + name] = (float)v.GetDouble();
                }
            }
        }

        return md;
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        var talker = c.Root.GetProperty("talker_config");
        var talkerTensors = new Dictionary<string, SafetensorsTensor>(StringComparer.Ordinal);
        var predictorTensors = new Dictionary<string, SafetensorsTensor>(StringComparer.Ordinal);
        var collector = new HfWeightCollector();
        string a = Architecture + ".";
        foreach (var (name, tensor) in tensors)
        {
            if (name == TalkerModel + "codec_embedding.weight")
            {
                talkerTensors["model.embed_tokens.weight"] = tensor;
            }
            else if (name == TalkerModel + "text_embedding.weight")
            {
                collector.Add(HfTransforms.Plain(tensor.ToWeightInfo(), a + "text_embedding.weight"));
            }
            else if (name == Talker + "codec_head.weight")
            {
                talkerTensors["lm_head.weight"] = tensor;
            }
            else if (name.StartsWith(TalkerModel, StringComparison.Ordinal))
            {
                talkerTensors["model." + name[TalkerModel.Length..]] = tensor;
            }
            else if (name.StartsWith(Talker + "text_projection.", StringComparison.Ordinal))
            {
                collector.Add(HfTransforms.Plain(tensor.ToWeightInfo(), a + name[Talker.Length..]));
            }
            else if (name.StartsWith(CodePredictor + "model.codec_embedding.", StringComparison.Ordinal))
            {
                collector.Add(HfTransforms.Plain(tensor.ToWeightInfo(), a + "code_predictor.embedding." + name[(CodePredictor + "model.codec_embedding.").Length..]));
            }
            else if (name.StartsWith(CodePredictor + "lm_head.", StringComparison.Ordinal))
            {
                collector.Add(HfTransforms.Plain(tensor.ToWeightInfo(), a + "code_predictor.head." + name[(CodePredictor + "lm_head.").Length..]));
            }
            else if (name.StartsWith(CodePredictor + "small_to_mtp_projection.", StringComparison.Ordinal))
            {
                collector.Add(HfTransforms.Plain(tensor.ToWeightInfo(), a + "code_predictor.projection." + name[(CodePredictor + "small_to_mtp_projection.").Length..]));
            }
            else if (name.StartsWith(CodePredictor + "model.", StringComparison.Ordinal))
            {
                predictorTensors["model." + name[(CodePredictor + "model.").Length..]] = tensor;
            }
            else
            {
                collector.AddUnmapped(name);
            }
        }

        // The code predictor's decoder takes embedding rows only; its first head stands in for the vocabulary (the host applies the head of each step).
        predictorTensors["model.embed_tokens.weight"] = tensors[CodePredictor + "model.codec_embedding.0.weight"];
        predictorTensors["lm_head.weight"] = tensors[CodePredictor + "lm_head.0.weight"];
        var mapping = new DecoderHfMapping();
        using (var config = Qwen3Config(talker, c.Directory))
        {
            foreach (var (_, w) in mapping.MapWeights(config, talkerTensors, options))
            {
                collector.Add(w);
            }
        }

        using (var config = Qwen3Config(talker.GetProperty("code_predictor_config"), c.Directory))
        {
            foreach (var (name, w) in mapping.MapWeights(config, predictorTensors, options))
            {
                // In f32: the talker's hidden state it reads has outliers near 100, beyond what reduced-precision activation paths for 16-bit weights keep exact.
                collector.Add(w.Type == DataType.BF16 || w.Type == DataType.F16 ? HfTransforms.ToF32(w, "code_predictor." + name) : w.As("code_predictor." + name));
            }
        }

        collector.ThrowIfUnmapped(c.ModelType);
        return collector.Weights;
    }

    /// <summary>The speech tokenizer folder as the vocoder companion.</summary>
    /// <param name="directory">The model folder.</param>
    public IReadOnlyDictionary<string, string> FindCompanions(string directory)
    {
        string codec = Path.Combine(directory, SpeechTokenizerFolder);
        return File.Exists(Path.Combine(codec, "config.json"))
            ? new Dictionary<string, string> { [AuxiliaryRoles.Vocoder] = codec }
            : new Dictionary<string, string>();
    }

    // A Qwen3 causal-LM configuration for one of the two decoders (the vocabulary is the codec's).
    private static HfConfig Qwen3Config(JsonElement source, string directory)
    {
        var node = JsonNode.Parse(source.GetRawText())!.AsObject();
        node["model_type"] = "qwen3";
        node["architectures"] = new JsonArray("Qwen3ForCausalLM");
        node["tie_word_embeddings"] = false;
        node.Remove("rope_scaling");
        node.Remove("code_predictor_config");
        node.Remove("layer_types");
        return HfConfig.Parse(node.ToJsonString(), directory);
    }
}
