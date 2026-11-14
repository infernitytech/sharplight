using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>k2-horizon</c> (K2-Horizon): RMS norms over <c>attention.group_norm_groups</c> slices of the hidden size, optional per-head q/k
/// norms (stored flat, one weight per head), an optional softplus output gate <c>attn_gate</c> with β = ln 2, leading dense layers and MoE layers
/// (experts routed by <c>expert_gating_func</c>, sigmoid by default, with the selection bias <c>exp_probs_b</c> and a shared expert). With
/// <c>attention.value_expert_count</c> the MoE layers compute their values with routed value experts (MoVA: router <c>attn_v_gate</c> whose
/// bias only selects, experts <c>attn_v_exps</c>), weighted like the experts. Following the reference implementation, the value expert weights
/// are renormalized whenever more than one is picked (llama.cpp renormalizes them only with <c>expert_weights_norm</c>).
/// </summary>
internal sealed class K2HorizonGgufMapping : GgufDecoderFamilyMapping, IGgufWeightTransformer
{
    private const string ValueExperts = "attention.value_experts.";

    // log2(1 + 2^x) of llama.cpp's gate: softplus with β = ln 2.
    private static readonly float s_gateBeta = MathF.Log(2f);

    private static readonly Dictionary<string, string> s_tensors = new(StringComparer.Ordinal)
    {
        ["attn_gate.weight"] = "attention.gate.weight",
        ["attn_v_gate.weight"] = ValueExperts + "router.weight",
        ["attn_v_gate.bias"] = ValueExperts + "router_bias",
        ["attn_v_exps.weight"] = ValueExperts + "weight",
    };

    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["k2-horizon"];

    /// <inheritdoc/>
    protected override bool NormalizesExpertWeights => false;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string>? LayerTensorOverrides => s_tensors;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        const string a = GgufDecoderHyperParameters.Prefix;
        if (g.Has("rope.dimension_count") && g.Int("rope.dimension_count") != HeadDimOf(g))
        {
            // The reference rotates channel pairs (i, head_dim/2 + i), llama.cpp pairs (i, rope_dim/2 + i): only full rotation agrees.
            throw new NotSupportedException("k2-horizon: a rotary width smaller than the head size is not supported.");
        }

        string activation = DecoderGgufMapping.RouterActivation(g.Int("expert_gating_func", 0)) ?? "sigmoid";
        if (g.Int("expert_count", 0) > 0)
        {
            md[a + "moe.router_activation"] = activation;
        }

        if (g.Int("attention.group_norm_groups", 1) is > 1 and var groups)
        {
            md[a + "norm.groups"] = groups;
        }

        md[a + "attention.gate_activation"] = "softplus";
        md[a + "attention.gate_softplus_beta"] = s_gateBeta;
        if (g.Int("attention.value_expert_count", 0) is > 0 and var valueExperts)
        {
            md[a + ValueExperts + "count"] = valueExperts;
            md[a + ValueExperts + "used_count"] = g.Int("attention.value_expert_used_count");
            md[a + ValueExperts + "router_activation"] = activation;
            if (g.Float("expert_weights_scale", 0f) is var scale && scale != 0f)
            {
                md[a + ValueExperts + "weights_scale"] = scale;
            }
        }
    }

    /// <summary>Views the flat per-head q/k norm weights as [heads, head size].</summary>
    public void Transform(Dictionary<string, WeightInfo> weights, GgufMetadataView gguf)
    {
        int heads = gguf.Int("attention.head_count");
        int kvHeads = gguf.Int("attention.head_count_kv", heads);
        foreach (var (name, w) in weights.ToArray())
        {
            int groups = name.EndsWith("attention.q_norm.weight", StringComparison.Ordinal) ? heads
                : name.EndsWith("attention.k_norm.weight", StringComparison.Ordinal) ? kvHeads : 0;
            if (groups > 0 && w.Shape.Rank == 1 && w.Shape.Elements() > HeadDimOf(gguf))
            {
                weights[name] = w.As(name, new Shape(groups, w.Shape.Elements() / groups));
            }
        }
    }
}
