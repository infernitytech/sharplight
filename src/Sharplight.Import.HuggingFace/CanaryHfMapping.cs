using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Hugging Face Canary checkpoints (<c>canary</c>: nvidia/canary-1b-v2 in transformers format) → canonical <c>encoder_decoder</c> with a
/// FastConformer encoder (<c>encoder.type</c> "conformer", mapped as for Parakeet, <see cref="ParakeetHfMapping"/>) and a pre-norm
/// transformer decoder (LayerNorms with biases, ReLU MLPs, NeMo's fixed sinusoidal positions as the learned-position table, scaled by
/// 1 / √hidden, and an embedding LayerNorm). The decoder is prompted with Canary's fixed template (start of context, start of transcript,
/// emotion, source and target language, punctuation, no inverse text normalization, no timestamps, no diarization: the chat template of
/// the checkpoint), recorded as <see cref="SpeechKeys.PromptTemplateIds"/>; requests without a language are English.
/// </summary>
internal sealed partial class CanaryHfMapping : IHfArchitectureMapping
{
    /// <summary>The template after the start token, by token text; {source} and {target} are the language tokens.</summary>
    private static readonly string[] s_template =
        ["<|startoftranscript|>", "<|emo:undefined|>", "{source}", "{target}", "<|pnc|>", "<|noitn|>", "<|notimestamp|>", "<|nodiarize|>"];

    private static readonly Dictionary<string, string> s_layerModules = new(StringComparer.Ordinal)
    {
        ["self_attn.q_proj"] = "attention.q",
        ["self_attn.k_proj"] = "attention.k",
        ["self_attn.v_proj"] = "attention.v",
        ["self_attn.o_proj"] = "attention.o",
        ["encoder_attn.q_proj"] = "cross_attention.q",
        ["encoder_attn.k_proj"] = "cross_attention.k",
        ["encoder_attn.v_proj"] = "cross_attention.v",
        ["encoder_attn.o_proj"] = "cross_attention.o",
        ["input_layernorm"] = "attention_norm",
        ["post_attention_layernorm"] = "cross_attention_norm",
        ["final_layernorm"] = "mlp_norm",
        ["mlp.fc1"] = "mlp.up",
        ["mlp.fc2"] = "mlp.down",
    };

    /// <inheritdoc/>
    public string Architecture => "encoder_decoder";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["canary"];

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig config)
    {
        foreach (string reason in ParakeetHfMapping.ValidateEncoder(config, "canary"))
        {
            yield return reason;
        }

        if (!config.Root.TryGetProperty("decoder_config", out var decoder) || decoder.ValueKind != JsonValueKind.Object)
        {
            yield return "canary: no decoder_config.";
            yield break;
        }

        if (Int(decoder, "hidden_size", 0) != Int(config.Root.GetProperty("encoder_config"), "hidden_size", 0))
        {
            yield return "canary: encoder and decoder widths differ (an encoder projection is not supported).";
        }

        if (Int(decoder, "num_key_value_heads", Int(decoder, "num_attention_heads", 8)) != Int(decoder, "num_attention_heads", 8))
        {
            yield return "canary: grouped key-value heads are not supported.";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        string a = Architecture + ".";
        var result = ParakeetHfMapping.EncoderHyperParameters(c, a);
        var decoder = c.Root.GetProperty("decoder_config");
        int hidden = Int(decoder, "hidden_size", 1024);
        int heads = Int(decoder, "num_attention_heads", 8);
        int headDim = Int(decoder, "head_dim", hidden / heads);
        result[a + "encoder.type"] = "conformer";
        result[a + "encoder.input"] = "audio";
        result[a + "decoder.layer_count"] = Int(decoder, "num_hidden_layers", 4);
        result[a + "attention.head_count"] = heads;
        result[a + "attention.head_dim"] = headDim;
        result[a + "attention.scale"] = 1f / MathF.Sqrt(headDim);
        result[a + "norm"] = "layer";
        result[a + "norm_eps"] = 1e-5f;
        result[a + "pre_norm"] = true;
        result[a + "mlp.activation"] = decoder.TryGetProperty("hidden_act", out var act) && act.GetString() is { } name ? name : "relu";
        result[a + "decoder_start_token"] = c.Int("decoder_start_token_id");
        result[a + "context_length"] = Int(decoder, "max_position_embeddings", 1024);
        result[a + "cross_attention.cached"] = true;
        return result;
    }

    [GeneratedRegex(@"^decoder\.layers\.(\d+)\.(.+)\.(weight|bias)$")]
    private static partial Regex LayerName();

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        var collector = new HfWeightCollector();
        var decoder = c.Root.GetProperty("decoder_config");
        int hidden = Int(decoder, "hidden_size", 1024);
        int positions = Int(decoder, "max_position_embeddings", 1024);
        collector.Add(WeightInfo.Derived("decoder.position_embedding.weight", DataType.F32, new Shape(positions, hidden),
            () => HfTransforms.EncodeF32(PositionTable(positions, hidden))));
        foreach (var (rawName, tensor) in tensors)
        {
            var info = tensor.ToWeightInfo();
            string name = rawName.StartsWith("model.", StringComparison.Ordinal) ? rawName["model.".Length..] : rawName;
            if (ParakeetHfMapping.TryMapEncoder(name, info, collector))
            {
                continue;
            }

            switch (name)
            {
                case "decoder.embed_tokens.weight":
                    collector.Add(info.As("token_embedding.weight"));
                    continue;
                case "proj_out.weight" or "proj_out.bias":
                    collector.Add(HfTransforms.Plain(info, "output." + name[(name.LastIndexOf('.') + 1)..]));
                    continue;
                case "decoder.embedding_layernorm.weight" or "decoder.embedding_layernorm.bias":
                    collector.Add(HfTransforms.Plain(info, "decoder.embedding_norm." + name[(name.LastIndexOf('.') + 1)..]));
                    continue;
                case "decoder.norm.weight" or "decoder.norm.bias":
                    collector.Add(HfTransforms.Plain(info, "decoder.output_norm." + name[(name.LastIndexOf('.') + 1)..]));
                    continue;
            }

            var m = LayerName().Match(name);
            if (!m.Success || !s_layerModules.TryGetValue(m.Groups[2].Value, out var module))
            {
                throw new NotSupportedException($"canary: tensor '{rawName}' has no canonical mapping.");
            }

            collector.Add(HfTransforms.Plain(info, $"decoder.layers.{m.Groups[1].Value}.{module}.{m.Groups[3].Value}"));
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

        int Token(string text) => ids.TryGetValue(text, out int id) ? id : throw new NotSupportedException($"canary: the vocabulary has no '{text}'.");
        int start = config.Int("decoder_start_token_id");
        metadata[SpeechKeys.StartId] = start;
        metadata[SpeechKeys.EndId] = config.Int("eos_token_id");
        metadata[SpeechKeys.PromptTemplateIds] = (int[])[start, .. s_template.Select(piece => piece switch
        {
            "{source}" => -1,
            "{target}" => -2,
            _ => Token(piece),
        })];
        var languages = ids.Where(e => e.Key.Length == 6 && e.Key.StartsWith("<|", StringComparison.Ordinal) && e.Key.EndsWith("|>", StringComparison.Ordinal)
                && char.IsAsciiLetterLower(e.Key[2]) && char.IsAsciiLetterLower(e.Key[3]))
            .OrderBy(e => e.Value)
            .ToArray();
        metadata[SpeechKeys.Languages] = languages.Select(e => e.Key[2..4]).ToArray();
        metadata[SpeechKeys.LanguageIds] = languages.Select(e => e.Value).ToArray();
        metadata[SpeechKeys.DefaultLanguage] = "en";
        metadata[SpeechKeys.MaxTextTokens] = 512;
    }

    private static int Int(JsonElement element, string key, int fallback)
        => element.TryGetProperty(key, out var value) && value.ValueKind == JsonValueKind.Number ? value.GetInt32() : fallback;

    // NeMo's FixedPositionalEncoding: [sin | cos] of the position over geometric timescales up to 10000^((C − 2) / C), divided by √C.
    private static float[] PositionTable(int length, int channels)
    {
        int half = channels / 2;
        double increment = Math.Log(Math.Pow(10000, (channels - 2.0) / channels)) / (half - 1);
        double scale = 1 / Math.Sqrt(channels);
        var table = new float[length * channels];
        for (int position = 0; position < length; position++)
        {
            for (int i = 0; i < half; i++)
            {
                // transformers multiplies in single precision; the angle's rounding matters at large positions.
                float angle = position * (float)Math.Exp(-increment * (float)i);
                table[(position * channels) + i] = (float)(Math.Sin(angle) * scale);
                table[(position * channels) + half + i] = (float)(Math.Cos(angle) * scale);
            }
        }

        return table;
    }
}
