using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Hugging Face Cohere Transcribe checkpoints (<c>cohere_asr</c>: CohereLabs/cohere-transcribe-03-2026; NeMo tensor names, read by transformers'
/// native <c>CohereAsrForConditionalGeneration</c> through its weight renaming) → canonical <c>encoder_decoder</c> with a FastConformer encoder
/// (<c>encoder.type</c> "conformer", the Parakeet encoder of <see cref="ParakeetHfMapping"/>, here 1280 wide), a linear projection of the encoder
/// output to the decoder width (<c>encoder.projection</c>) and a pre-norm transformer decoder (LayerNorms with biases, ReLU MLPs, the checkpoint's
/// sinusoidal position table and an embedding LayerNorm), as Canary's (<see cref="CanaryHfMapping"/>). The prompt is the processor's template
/// (start "▁", start of context, start of transcript, emotion, source and target language, punctuation, no inverse text normalization, no
/// timestamps, no diarization) as <see cref="SpeechKeys.PromptTemplateIds"/>; the model cannot detect the language and defaults to English.
/// Recordings longer than 35 s are split on the waveform at the quietest 0.1 s of the last 5 s before the limit, each chunk with its own
/// features (<see cref="SpeechKeys.ChunkSeconds"/>), and Japanese and Chinese chunk texts join without a space, as transformers' feature extractor
/// and processor do. The feature extractor's deterministic dither (1e-5 Gaussian noise) is not reproduced.
/// </summary>
internal sealed partial class CohereAsrHfMapping : IHfArchitectureMapping
{
    /// <summary>The longest chunk of a recording, in seconds (the feature extractor's <c>max_audio_clip_s</c>).</summary>
    private const int ChunkSeconds = 35;

    /// <summary>The template after the start token, by token text; {source} and {target} are the language tokens.</summary>
    private static readonly string[] s_template =
        ["<|startofcontext|>", "<|startoftranscript|>", "<|emo:undefined|>", "{source}", "{target}", "<|pnc|>", "<|noitn|>", "<|notimestamp|>", "<|nodiarize|>"];

    /// <summary>The languages of transformers' processor (the checkpoint's <c>supported_languages</c> when it lists them).</summary>
    private static readonly string[] s_languages = ["ar", "de", "el", "en", "es", "fr", "it", "ja", "ko", "nl", "pl", "pt", "vi", "zh"];

    /// <summary>NeMo encoder module names → transformers' (which <see cref="ParakeetHfMapping.TryMapEncoder"/> maps).</summary>
    private static readonly (string From, string To)[] s_encoderRenames =
    [
        (".self_attn.linear_q.", ".self_attn.q_proj."),
        (".self_attn.linear_k.", ".self_attn.k_proj."),
        (".self_attn.linear_v.", ".self_attn.v_proj."),
        (".self_attn.linear_out.", ".self_attn.o_proj."),
        (".self_attn.linear_pos.", ".self_attn.relative_k_proj."),
        (".self_attn.pos_bias_u", ".self_attn.bias_u"),
        (".self_attn.pos_bias_v", ".self_attn.bias_v"),
        (".conv.batch_norm.", ".conv.norm."),
    ];

    private static readonly Dictionary<string, string> s_decoderModules = new(StringComparer.Ordinal)
    {
        ["first_sub_layer.query_net"] = "attention.q",
        ["first_sub_layer.key_net"] = "attention.k",
        ["first_sub_layer.value_net"] = "attention.v",
        ["first_sub_layer.out_projection"] = "attention.o",
        ["second_sub_layer.query_net"] = "cross_attention.q",
        ["second_sub_layer.key_net"] = "cross_attention.k",
        ["second_sub_layer.value_net"] = "cross_attention.v",
        ["second_sub_layer.out_projection"] = "cross_attention.o",
        ["layer_norm_1"] = "attention_norm",
        ["layer_norm_2"] = "cross_attention_norm",
        ["layer_norm_3"] = "mlp_norm",
        ["third_sub_layer.dense_in"] = "mlp.up",
        ["third_sub_layer.dense_out"] = "mlp.down",
    };

    /// <inheritdoc/>
    public string Architecture => "encoder_decoder";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["cohere_asr"];

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig config)
    {
        if (!config.Root.TryGetProperty("encoder", out var encoder) || encoder.ValueKind != JsonValueKind.Object)
        {
            yield return "cohere_asr: no encoder configuration.";
            yield break;
        }

        if (!config.Root.TryGetProperty("transf_decoder", out var wrapper) || !wrapper.TryGetProperty("config_dict", out var decoder) || decoder.ValueKind != JsonValueKind.Object)
        {
            yield return "cohere_asr: no transf_decoder.config_dict.";
            yield break;
        }

        if (Text(encoder, "self_attention_model", "rel_pos") != "rel_pos" || Text(encoder, "subsampling", "dw_striding") != "dw_striding"
            || Int(encoder, "subsampling_factor", 8) != 8 || Text(encoder, "conv_norm_type", "batch_norm") != "batch_norm")
        {
            yield return "cohere_asr: only the FastConformer with relative-position attention, 8× depthwise-striding subsampling and batch norm is supported.";
        }

        if (encoder.TryGetProperty("att_context_size", out var context) && context.ValueKind == JsonValueKind.Array && context.EnumerateArray().Any(e => e.GetInt32() != -1))
        {
            yield return "cohere_asr: limited attention context is not supported.";
        }

        if (encoder.TryGetProperty("xscaling", out var scaling) && scaling.ValueKind == JsonValueKind.True)
        {
            yield return "cohere_asr: input scaling (xscaling) is not supported.";
        }

        if (!decoder.TryGetProperty("pre_ln", out var preNorm) || preNorm.ValueKind != JsonValueKind.True)
        {
            yield return "cohere_asr: only the pre-norm decoder is supported.";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        string a = Architecture + ".";
        var encoder = c.Root.GetProperty("encoder");
        var decoder = c.Root.GetProperty("transf_decoder").GetProperty("config_dict");
        int encoderHidden = Int(encoder, "d_model", 1280);
        var result = ParakeetHfMapping.EncoderHyperParameters(c, a, encoderHidden, Int(encoder, "n_layers", 48), Int(encoder, "n_heads", 8),
            Int(encoder, "subsampling_conv_channels", 256), "silu", 1f, ChunkSeconds);
        int hidden = Int(decoder, "hidden_size", 1024);
        int heads = Int(decoder, "num_attention_heads", 8);
        int headDim = hidden / heads;
        result[a + "hidden_size"] = hidden;
        result[a + "encoder.hidden_size"] = encoderHidden;
        result[a + "encoder.type"] = "conformer";
        result[a + "encoder.input"] = "audio";
        result[a + "decoder.layer_count"] = Int(decoder, "num_layers", 8);
        result[a + "attention.head_count"] = heads;
        result[a + "attention.head_dim"] = headDim;
        result[a + "attention.scale"] = 1f / MathF.Sqrt(headDim);
        result[a + "norm"] = "layer";
        result[a + "norm_eps"] = 1e-5f;
        result[a + "pre_norm"] = true;
        result[a + "mlp.activation"] = Text(decoder, "hidden_act", "relu");
        result[a + "decoder_start_token"] = DecoderStart(c);
        result[a + "context_length"] = Int(decoder, "max_sequence_length", 1024);
        result[a + "cross_attention.cached"] = true;
        return result;
    }

    [GeneratedRegex(@"^transf_decoder\._decoder\.layers\.(\d+)\.(.+)\.(weight|bias)$")]
    private static partial Regex DecoderLayerName();

    [GeneratedRegex(@"^encoder\.pre_encode\.conv\.(\d+)\.(weight|bias)$")]
    private static partial Regex SubsamplingName();

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        var collector = new HfWeightCollector();
        foreach (var (name, tensor) in tensors)
        {
            var info = tensor.ToWeightInfo();
            if (name.StartsWith("preprocessor.", StringComparison.Ordinal))
            {
                // The exported filter bank and window: transformers computes its own (the slaney bank, a symmetric Hann window), as the importer does.
                continue;
            }

            if (name.StartsWith("encoder.", StringComparison.Ordinal) && ParakeetHfMapping.TryMapEncoder(EncoderName(name), info, collector))
            {
                continue;
            }

            string suffix = name[(name.LastIndexOf('.') + 1)..];
            switch (name)
            {
                case "encoder_decoder_proj.weight" or "encoder_decoder_proj.bias":
                    collector.Add(HfTransforms.Plain(info, "encoder.projection." + suffix));
                    continue;
                case "transf_decoder._embedding.token_embedding.weight":
                    collector.Add(info.As("token_embedding.weight"));
                    continue;
                case "transf_decoder._embedding.position_embedding.pos_enc":
                    collector.Add(info.As("decoder.position_embedding.weight"));
                    continue;
                case "transf_decoder._embedding.layer_norm.weight" or "transf_decoder._embedding.layer_norm.bias":
                    collector.Add(HfTransforms.Plain(info, "decoder.embedding_norm." + suffix));
                    continue;
                case "transf_decoder._decoder.final_layer_norm.weight" or "transf_decoder._decoder.final_layer_norm.bias":
                    collector.Add(HfTransforms.Plain(info, "decoder.output_norm." + suffix));
                    continue;
                case "log_softmax.mlp.layer0.weight" or "log_softmax.mlp.layer0.bias":
                    // The head ends in a log-softmax, which does not change the order of the logits.
                    collector.Add(HfTransforms.Plain(info, "output." + suffix));
                    continue;
            }

            var layer = DecoderLayerName().Match(name);
            if (!layer.Success || !s_decoderModules.TryGetValue(layer.Groups[2].Value, out var module))
            {
                throw new NotSupportedException($"cohere_asr: tensor '{name}' has no canonical mapping.");
            }

            collector.Add(HfTransforms.Plain(info, $"decoder.layers.{layer.Groups[1].Value}.{module}.{layer.Groups[3].Value}"));
        }

        return collector.Weights;
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, IWeightSource> MapAttachments(HfConfig config, string directory) => ParakeetHfMapping.FilterBank(config);

    /// <inheritdoc/>
    public void AddMetadata(IDictionary<string, object> metadata, HfConfig config, string directory)
    {
        var tokens = metadata.TryGetValue(CanonicalKeys.TokenizerTokens, out var t) && t is string[] list ? list : [];
        var ids = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < tokens.Length; i++)
        {
            ids.TryAdd(tokens[i], i);
        }

        int Token(string text) => ids.TryGetValue(text, out int id) ? id : throw new NotSupportedException($"cohere_asr: the vocabulary has no '{text}'.");
        int start = DecoderStart(config);
        metadata[SpeechKeys.StartId] = start;
        metadata[SpeechKeys.EndId] = Token("<|endoftext|>");
        metadata[SpeechKeys.PromptTemplateIds] = (int[])[start, .. s_template.Select(piece => piece switch
        {
            "{source}" => -1,
            "{target}" => -2,
            _ => Token(piece),
        })];
        string[] languages = config.Root.TryGetProperty("supported_languages", out var supported) && supported.ValueKind == JsonValueKind.Array
            ? [.. supported.EnumerateArray().Select(e => e.GetString()!).Order(StringComparer.Ordinal)]
            : s_languages;
        metadata[SpeechKeys.Languages] = languages;
        metadata[SpeechKeys.LanguageIds] = languages.Select(code => Token($"<|{code}|>")).ToArray();
        metadata[SpeechKeys.DefaultLanguage] = "en";
        metadata[SpeechKeys.MaxTextTokens] = 512;
        metadata[SpeechKeys.ChunkSeconds] = Number(config.Root, "max_audio_clip_s", ChunkSeconds);
        metadata[SpeechKeys.ChunkSearchSeconds] = Number(config.Root, "overlap_chunk_second", 5);
        metadata[SpeechKeys.ChunkEnergyWindow] = Int(config.Root, "min_energy_window_samples", 1600);
        metadata[SpeechKeys.UnspacedLanguages] = (string[])["ja", "zh"];
    }

    // The decoder's first token: generation_config.json's decoder_start_token_id ("▁", which transformers' processor puts first).
    private static int DecoderStart(HfConfig config)
    {
        if (config.GetInt("decoder_start_token_id") is int start)
        {
            return start;
        }

        string path = Path.Combine(config.Directory, "generation_config.json");
        if (File.Exists(path))
        {
            using var document = JsonDocument.Parse(File.ReadAllBytes(path));
            if (document.RootElement.TryGetProperty("decoder_start_token_id", out var value) && value.ValueKind == JsonValueKind.Number)
            {
                return value.GetInt32();
            }
        }

        throw new NotSupportedException("cohere_asr: no decoder_start_token_id in config.json or generation_config.json.");
    }

    // NeMo's encoder tensor names → transformers' (the subsampling stack keeps its module indices).
    private static string EncoderName(string name)
    {
        var subsampling = SubsamplingName().Match(name);
        if (subsampling.Success)
        {
            return $"encoder.subsampling.layers.{subsampling.Groups[1].Value}.{subsampling.Groups[2].Value}";
        }

        if (name.StartsWith("encoder.pre_encode.out.", StringComparison.Ordinal))
        {
            return "encoder.subsampling.linear." + name["encoder.pre_encode.out.".Length..];
        }

        foreach (var (from, to) in s_encoderRenames)
        {
            name = name.Replace(from, to, StringComparison.Ordinal);
        }

        return name;
    }

    private static int Int(JsonElement element, string key, int fallback)
        => element.TryGetProperty(key, out var value) && value.ValueKind == JsonValueKind.Number ? value.GetInt32() : fallback;

    private static double Number(JsonElement element, string key, double fallback)
        => element.TryGetProperty(key, out var value) && value.ValueKind == JsonValueKind.Number ? value.GetDouble() : fallback;

    private static string Text(JsonElement element, string key, string fallback)
        => element.TryGetProperty(key, out var value) && value.ValueKind == JsonValueKind.String ? value.GetString()! : fallback;
}
