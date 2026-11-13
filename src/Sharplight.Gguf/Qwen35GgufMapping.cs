using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>qwen35</c> (Qwen3.5 / 3.6 / 3.8 dense hybrid), <c>qwen35moe</c> (the MoE variants) and <c>qwen3next</c>
/// (Qwen3-Next / Qwen3-Coder-Next: plain partial RoPE, fused <c>ssm_ba</c>, value heads grouped by key head) → canonical
/// <c>qwen3_5</c>.
/// </summary>
internal sealed class Qwen35GgufMapping : IGgufArchitectureMapping, IGgufSplitMapping
{
    private const int RopeSectionCount = 4;
    private const string Qwen3NextArchitecture = "qwen3next";
    private const string SharedExpertGate = "mlp.shared.input_gate.weight";

    /// <inheritdoc/>
    public IReadOnlyList<string> GgufArchitectures { get; } = ["qwen35", "qwen35moe", Qwen3NextArchitecture];

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
            [a + "ffn_hidden_size"] = g.Int("feed_forward_length", g.Int("expert_feed_forward_length", 0)),
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
        if (g.Has("expert_count"))
        {
            md[a + "moe.expert_count"] = g.Int("expert_count");
            md[a + "moe.expert_used_count"] = g.Int("expert_used_count");
            md[a + "moe.normalize_weights"] = true;
        }

        if (IsQwen3Next(g))
        {
            md[a + "linear_attention.grouped_value_heads"] = true;
        }

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
        ["ffn_gate_inp.weight"] = "mlp.router.weight",
        ["ffn_gate_exps.weight"] = "mlp.experts.gate.weight",
        ["ffn_up_exps.weight"] = "mlp.experts.up.weight",
        ["ffn_gate_up_exps.weight"] = "mlp.experts.gate_up.weight",
        ["ffn_down_exps.weight"] = "mlp.experts.down.weight",
        ["ffn_gate_shexp.weight"] = "mlp.shared.gate.weight",
        ["ffn_up_shexp.weight"] = "mlp.shared.up.weight",
        ["ffn_down_shexp.weight"] = "mlp.shared.down.weight",
        ["ffn_gate_inp_shexp.weight"] = SharedExpertGate,
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

    private static bool IsQwen3Next(GgufMetadataView g) => g.GgufArchitecture == Qwen3NextArchitecture;

    /// <inheritdoc/>
    public WeightInfo MapWeight(WeightInfo weight, GgufMetadataView gguf)
    {
        // The shared expert's sigmoid gate is a vector in GGUF; canonically it is a [1, hidden] projection.
        if (weight.Name.EndsWith(SharedExpertGate, StringComparison.Ordinal) && weight.Shape.Rank == 1)
        {
            return weight.As(weight.Name, new Shape(1, weight.Shape[0]));
        }

        return weight;
    }

    /// <summary>
    /// qwen3next keeps the HF <c>in_proj_ba</c> layout: per key head [beta (r rows), alpha (r rows)] with r value heads per key
    /// head; it splits into the canonical beta and alpha projections (value heads grouped by key head).
    /// </summary>
    public IReadOnlyList<WeightInfo>? Split(string ggufName, WeightInfo tensor, GgufMetadataView gguf)
    {
        if (!ggufName.EndsWith(".ssm_ba.weight", StringComparison.Ordinal)
            || MapWeightName(ggufName.Replace(".ssm_ba.", ".ssm_beta.", StringComparison.Ordinal), gguf) is not { } betaName)
        {
            return null;
        }

        int keyHeads = gguf.Int("ssm.group_count");
        int perKeyHead = gguf.Int("ssm.time_step_rank") / keyHeads;
        var starts = Enumerable.Range(0, keyHeads).Select(h => (long)h * 2 * perKeyHead).ToArray();
        string alphaName = betaName.Replace("linear_attention.beta.", "linear_attention.alpha.", StringComparison.Ordinal);
        return
        [
            tensor.GatherRows(betaName, starts.Select(s => (s, (long)perKeyHead))),
            tensor.GatherRows(alphaName, starts.Select(s => (s + perKeyHead, (long)perKeyHead))),
        ];
    }

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
