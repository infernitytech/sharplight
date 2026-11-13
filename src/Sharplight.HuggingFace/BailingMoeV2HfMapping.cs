using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>bailing_moe_v2</c> (Ling / Ring 2.0, remote code): a fused q|k|v projection (<c>attention.query_key_value</c>),
/// per-head q/k norms before a partial rotary embedding (<c>partial_rotary_factor</c>, default one half of the head), leading dense
/// layers and a sigmoid router with selection bias (<c>gate.expert_bias</c>), group-limited top-k, renormalized and scaled weights
/// plus shared experts.
/// </summary>
internal sealed class BailingMoeV2HfMapping : HfMoeFamilyMapping
{
    private const double DefaultRotaryFraction = 0.5;

    private static readonly IReadOnlyDictionary<string, string> s_layerTensors = With(LlamaLayerTensors,
        ("attention.dense.weight", "attention.o.weight"),
        ("attention.dense.bias", "attention.o.bias"),
        ("attention.query_layernorm.weight", "attention.q_norm.weight"),
        ("attention.key_layernorm.weight", "attention.k_norm.weight"));

    private static readonly IReadOnlyDictionary<string, string> s_globalTensors = With(LlamaGlobalTensors,
        ("model.word_embeddings.weight", "token_embedding.weight"));

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["bailing_moe_v2"];

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> LayerTensors => s_layerTensors;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> GlobalTensors => s_globalTensors;

    /// <inheritdoc/>
    protected override int RopeDimension(HfConfig c, int headDim)
        => (int)(headDim * (c.RopeDouble("partial_rotary_factor", "partial_rotary_factor") ?? DefaultRotaryFraction));

    /// <inheritdoc/>
    protected override MoeRouting Routing(HfConfig c)
        => new(c.Int("num_experts"), c.Int("num_experts_per_tok"), c.Int("moe_intermediate_size"),
            c.GetString("score_function") == "softmax" ? "softmax" : "sigmoid", c.GetBool("norm_topk_prob") ?? true,
            (float)(c.GetDouble("routed_scaling_factor") ?? 1.0), c.GetInt("n_group") ?? 1, c.GetInt("topk_group") ?? 1);

    /// <inheritdoc/>
    protected override bool TryMapFamilyTensor(HfConfig c, string layerPrefix, string suffix, WeightInfo w, HfWeightCollector collector)
    {
        if (suffix is not ("attention.query_key_value.weight" or "attention.query_key_value.bias"))
        {
            return false;
        }

        string kind = suffix.EndsWith(".bias", StringComparison.Ordinal) ? "bias" : "weight";
        int headDim = HeadDimOf(c);
        int heads = HeadCount(c);
        int kvHeads = KvHeadCount(c, heads);
        int q = heads * headDim;
        int kv = kvHeads * headDim;
        collector.Add(HfFusedProjections.GatherRows(w, $"{layerPrefix}attention.q.{kind}", [.. Enumerable.Range(0, q)]));
        collector.Add(HfFusedProjections.GatherRows(w, $"{layerPrefix}attention.k.{kind}", [.. Enumerable.Range(q, kv)]));
        collector.Add(HfFusedProjections.GatherRows(w, $"{layerPrefix}attention.v.{kind}", [.. Enumerable.Range(q + kv, kv)]));
        return true;
    }
}
