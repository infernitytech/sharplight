namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>eagle3</c> drafters → canonical <c>eagle3</c> (vocabulary on <c>Eagle3DraftHead</c>): one decoder layer over the fused
/// features of three target layers (<c>target_layers</c>, the inputs of those layers), with the target's tokenizer. The converter permutes q and
/// k for llama.cpp's interleaved rope pairs, stores <c>d2t</c> as absolute target token ids and the optional feature norm as
/// <c>enc.output_norm</c>; <c>rope_freqs</c> carries llama3 scaling.
/// </summary>
internal sealed class Eagle3GgufMapping : IGgufArchitectureMapping
{
    /// <inheritdoc/>
    public IReadOnlyList<string> GgufArchitectures { get; } = ["eagle3"];

    /// <inheritdoc/>
    public string Architecture => "eagle3";

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(GgufMetadataView g)
    {
        string a = Architecture + ".";
        int hidden = g.Int("embedding_length");
        int heads = g.Int("attention.head_count");
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "hidden_size"] = hidden,
            [a + "head_count"] = heads,
            [a + "kv_head_count"] = g.Int("attention.head_count_kv", heads),
            [a + "head_dim"] = g.Int("attention.key_length", hidden / heads),
            [a + "norm_eps"] = g.Float("attention.layer_norm_rms_epsilon"),
            [a + "rope.base"] = g.Float("rope.freq_base", GgufMappingHelpers.DefaultRopeBase),
            [a + "rope.style"] = "gptj",
            [a + "target_layers"] = ((Array)(g.Raw("target_layers") ?? throw new InvalidDataException("eagle3: target_layers is missing.")))
                .Cast<object>().Select(Convert.ToInt32).ToArray(),
            [a + "target_hidden_size"] = g.Int("target_hidden_size"),
            [a + "norm_before_residual"] = g.Bool("norm_before_residual", false),
            [a + "norm_before_fc"] = g.Bool("norm_before_fc", false),
        };
        GgufMappingHelpers.AddContextLength(md, a, g);
        return md;
    }

    private static readonly Dictionary<string, string> s_global = new(StringComparer.Ordinal)
    {
        ["token_embd.weight"] = "token_embedding.weight",
        ["output_norm.weight"] = "output_norm.weight",
        ["output.weight"] = "lm_head.weight",
        ["rope_freqs.weight"] = "rope.freq_factors",
        ["fc.weight"] = "fc.weight",
        ["enc.output_norm.weight"] = "fc_norm.weight",
        ["d2t"] = "draft_to_target",
    };

    private static readonly Dictionary<string, string> s_layer = new(StringComparer.Ordinal)
    {
        ["attn_norm.weight"] = "input_norm.weight",
        ["attn_norm_2.weight"] = "hidden_norm.weight",
        ["attn_q.weight"] = "attention.q.weight",
        ["attn_k.weight"] = "attention.k.weight",
        ["attn_v.weight"] = "attention.v.weight",
        ["attn_output.weight"] = "attention.o.weight",
        ["ffn_norm.weight"] = "post_attention_norm.weight",
        ["ffn_gate.weight"] = "mlp.gate.weight",
        ["ffn_up.weight"] = "mlp.up.weight",
        ["ffn_down.weight"] = "mlp.down.weight",
    };

    /// <inheritdoc/>
    public string? MapWeightName(string name, GgufMetadataView g)
    {
        if (s_global.TryGetValue(name, out var global))
        {
            return global;
        }

        var m = GgufTensorNames.Block().Match(name);
        return m.Success && m.Groups[1].Value == "0" && s_layer.TryGetValue(m.Groups[2].Value, out var layer) ? "layers.0." + layer : null;
    }
}
