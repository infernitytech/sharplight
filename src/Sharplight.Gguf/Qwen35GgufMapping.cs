namespace Sharplight.Gguf;

/// <summary>llama.cpp <c>qwen35</c> (Qwen3.5 / 3.6 / 3.8 dense hybrid) → canonical <c>qwen3_5</c>.</summary>
internal sealed class Qwen35GgufMapping : IGgufArchitectureMapping
{
    private const int RopeSectionCount = 4;

    /// <inheritdoc/>
    public IReadOnlyList<string> GgufArchitectures { get; } = ["qwen35"];

    /// <inheritdoc/>
    public string Architecture => "qwen3_5";

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(GgufMetadataView g)
    {
        int nextnLayers = g.Int("nextn_predict_layers", 0);
        int valueHeads = g.Int("ssm.time_step_rank");
        string a = Architecture + ".";
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = g.Int("block_count") - nextnLayers,
            [a + "hidden_size"] = g.Int("embedding_length"),
            [a + "ffn_hidden_size"] = g.Int("feed_forward_length"),
            [a + "norm_eps"] = g.Float("attention.layer_norm_rms_epsilon"),
            [a + "attention.head_count"] = g.Int("attention.head_count"),
            [a + "attention.kv_head_count"] = g.Int("attention.head_count_kv"),
            [a + "attention.head_dim"] = g.Int("attention.key_length"),
            [a + "rope.dim"] = g.Int("rope.dimension_count"),
            [a + "full_attention_interval"] = g.Int("full_attention_interval", 4),
            [a + "linear_attention.conv_kernel"] = g.Int("ssm.conv_kernel"),
            [a + "linear_attention.key_head_dim"] = g.Int("ssm.state_size"),
            [a + "linear_attention.key_head_count"] = g.Int("ssm.group_count"),
            [a + "linear_attention.value_head_count"] = valueHeads,
            [a + "linear_attention.value_head_dim"] = g.Int("ssm.inner_size") / valueHeads,
            [a + "mtp.layer_count"] = nextnLayers,
        };
        GgufMappingHelpers.AddContextLength(md, a, g);

        if (g.Has("rope.freq_base"))
        {
            md[a + "rope.base"] = g.Float("rope.freq_base");
        }

        if (g.Raw("rope.dimension_sections") is Array sections)
        {
            var canonicalSections = new int[RopeSectionCount];
            for (int i = 0; i < Math.Min(RopeSectionCount, sections.Length); i++)
            {
                canonicalSections[i] = Convert.ToInt32(sections.GetValue(i));
            }

            md[a + "rope.sections"] = canonicalSections;
        }

        return md;
    }

    /// <summary>Per-layer tensors (shared by the trunk and the MTP layers), GGUF suffix to canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_layer = new(StringComparer.Ordinal)
    {
        ["attn_norm.weight"] = "input_norm.weight",
        ["post_attention_norm.weight"] = "post_attention_norm.weight",
        ["attn_q.weight"] = "attention.q.weight",
        ["attn_k.weight"] = "attention.k.weight",
        ["attn_v.weight"] = "attention.v.weight",
        ["attn_output.weight"] = "attention.o.weight",
        ["attn_q_norm.weight"] = "attention.q_norm.weight",
        ["attn_k_norm.weight"] = "attention.k_norm.weight",
        ["attn_qkv.weight"] = "linear_attention.qkv.weight",
        ["attn_gate.weight"] = "linear_attention.z.weight",
        ["ssm_beta.weight"] = "linear_attention.beta.weight",
        ["ssm_alpha.weight"] = "linear_attention.alpha.weight",
        ["ssm_out.weight"] = "linear_attention.out.weight",
        ["ssm_norm.weight"] = "linear_attention.norm.weight",
        ["ssm_conv1d.weight"] = "linear_attention.conv1d.weight",
        ["ssm_a"] = "linear_attention.a",
        ["ssm_dt.bias"] = "linear_attention.dt_bias",
        ["ffn_gate.weight"] = "mlp.gate.weight",
        ["ffn_up.weight"] = "mlp.up.weight",
        ["ffn_down.weight"] = "mlp.down.weight",
    };

    /// <summary>Tensors shared by all MTP (NextN) layers, GGUF suffix to the canonical name below <c>mtp.</c>.</summary>
    private static readonly Dictionary<string, string> s_nextn = new(StringComparer.Ordinal)
    {
        ["nextn.eh_proj.weight"] = "fc.weight",
        ["nextn.enorm.weight"] = "embedding_norm.weight",
        ["nextn.hnorm.weight"] = "hidden_norm.weight",
        ["nextn.shared_head_norm.weight"] = "output_norm.weight",
        ["nextn.shared_head_head.weight"] = "lm_head.weight",
        ["nextn.embed_tokens.weight"] = "token_embedding.weight",
    };

    /// <inheritdoc/>
    public string? MapWeightName(string name, GgufMetadataView g)
    {
        if (GgufTensorNames.MapGlobal(name) is { } globalName)
        {
            return globalName;
        }

        var m = GgufTensorNames.Block().Match(name);
        if (!m.Success)
        {
            return null;
        }

        int layer = int.Parse(m.Groups[1].Value);
        string suffix = m.Groups[2].Value;
        int trunkLayers = g.Int("block_count") - g.Int("nextn_predict_layers", 0);
        if (layer < trunkLayers)
        {
            return s_layer.TryGetValue(suffix, out var trunkName) ? $"layers.{layer}.{trunkName}" : null;
        }

        // MTP (NextN) layers: per-layer tensors under mtp.layers.{j}, the shared ones under mtp.*.
        if (s_nextn.TryGetValue(suffix, out var sharedName))
        {
            return $"mtp.{sharedName}";
        }

        return s_layer.TryGetValue(suffix, out var mtpName) ? $"mtp.layers.{layer - trunkLayers}.{mtpName}" : null;
    }
}
