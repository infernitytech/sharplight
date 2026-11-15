using System.Runtime.CompilerServices;
using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Hugging Face <c>k2_horizon</c> (K2-Horizon, remote code): RMS norms over <c>layernorm_num_groups</c> slices of the hidden size, optional per-head
/// q/k norms (one weight per head, stored flat), an optional softplus output gate with β = ln 2 (<c>attention_gate_func</c> "softplus"), and
/// dense layers (<c>mlp_only_layers</c>) next to MoE layers with sigmoid or softmax experts, a selection bias (<c>mlp.gate.bias</c>),
/// renormalization (<c>norm_topk_prob</c>), <c>router_scaling_factor</c> and a shared expert. With <c>mova_num_experts</c> the MoE layers
/// compute their attention values with routed value experts (MoVA: <c>self_attn.v_router</c> with its selection bias, <c>self_attn.v_experts.{j}</c>),
/// weighted like the experts but always renormalized when more than one is picked.
/// </summary>
internal sealed partial class K2HorizonHfMapping : HfMoeFamilyMapping
{
    private const string ValueExperts = "attention.value_experts.";

    // softplus(x, beta = ln 2) of the remote code's attention gate.
    private static readonly float s_gateBeta = MathF.Log(2f);

    private static readonly IReadOnlyDictionary<string, string> s_layerTensors = With(LlamaLayerTensors,
        ("self_attn.gate_proj.weight", "attention.gate.weight"),
        ("self_attn.v_router.weight", ValueExperts + "router.weight"));

    // Value expert tensors of one import, keyed by that import's collector (the mapping itself is shared).
    private static readonly ConditionalWeakTable<HfWeightCollector, HfPartJoiner> s_valueExperts = [];

    [GeneratedRegex(@"^self_attn\.v_experts\.(\d+)\.weight$")]
    private static partial Regex ValueExpertName();

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["k2_horizon"];

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> LayerTensors => s_layerTensors;

    /// <inheritdoc/>
    protected override int RopeDimension(HfConfig c, int headDim)
    {
        if (c.GetInt("rope_head_dim") is { } ropeDim && ropeDim != headDim)
        {
            // The remote code rotates channel pairs (i, head_dim/2 + i) for i < rope_head_dim/2, which needs permuted q/k rows.
            throw new NotSupportedException("k2_horizon: rope_head_dim smaller than head_dim is not supported.");
        }

        return headDim;
    }

    /// <inheritdoc/>
    protected override MoeRouting? Routing(HfConfig c)
        => c.GetInt("num_experts") is > 0 and var experts
            ? new(experts, c.Int("num_experts_per_tok"), c.Int("moe_intermediate_size"), ScoreFunction(c), c.GetBool("norm_topk_prob") ?? false,
                RoutedScale(c))
            : null;

    /// <inheritdoc/>
    protected override void AddFamilyHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        if (c.GetBool("use_sliding_window") == true)
        {
            throw new NotSupportedException("k2_horizon: use_sliding_window is not supported.");
        }

        if (c.GetInt("layernorm_num_groups") is > 1 and var groups)
        {
            md["decoder.norm.groups"] = groups;
        }

        switch (c.GetString("attention_gate_func"))
        {
            case null:
                break;
            case "softplus":
                md["decoder.attention.gate_activation"] = "softplus";
                md["decoder.attention.gate_softplus_beta"] = s_gateBeta;
                break;
            case var other:
                throw new NotSupportedException($"k2_horizon: attention_gate_func '{other}' is not supported.");
        }

        if (c.GetInt("mova_num_experts") is > 0 and var valueExperts)
        {
            md["decoder." + ValueExperts + "count"] = valueExperts;
            md["decoder." + ValueExperts + "used_count"] = c.Int("mova_num_experts_per_tok");
            md["decoder." + ValueExperts + "router_activation"] = ScoreFunction(c);
            md["decoder." + ValueExperts + "weights_scale"] = RoutedScale(c);
        }
    }

    /// <inheritdoc/>
    protected override bool TryMapFamilyTensor(HfConfig c, string layerPrefix, string suffix, WeightInfo w, HfWeightCollector collector)
    {
        switch (suffix)
        {
            case "mlp.gate.bias":
                collector.Add(HfTransforms.ToF32(w, layerPrefix + "mlp.router_bias"));
                return true;
            case "self_attn.v_router.bias":
                collector.Add(HfTransforms.ToF32(w, layerPrefix + ValueExperts + "router_bias"));
                return true;
            case "self_attn.q_norm.weight":
                collector.Add(PerHeadNorm(w, layerPrefix + "attention.q_norm.weight", c.Int("num_attention_heads")));
                return true;
            case "self_attn.k_norm.weight":
                collector.Add(PerHeadNorm(w, layerPrefix + "attention.k_norm.weight", c.Int("num_key_value_heads")));
                return true;
        }

        if (ValueExpertName().Match(suffix) is not { Success: true } match)
        {
            return false;
        }

        s_valueExperts.GetValue(collector, _ => HfPartJoiner.Experts()).Add(layerPrefix + ValueExperts + "weight", int.Parse(match.Groups[1].Value), w);
        return true;
    }

    /// <inheritdoc/>
    protected override void FinishWeights(HfConfig c, HfWeightCollector collector)
    {
        base.FinishWeights(c, collector);
        if (s_valueExperts.TryGetValue(collector, out var valueExperts))
        {
            valueExperts.JoinInto(collector, c.Int("mova_num_experts"));
            s_valueExperts.Remove(collector);
        }
    }

    private static string ScoreFunction(HfConfig c) => c.GetString("router_score_func") ?? "softmax";

    private static float RoutedScale(HfConfig c) => (float)(c.GetDouble("router_scaling_factor") ?? 1.0);

    // The q/k norm normalizes every head separately with that head's slice of the flat weight: canonical [heads, head size].
    private static WeightInfo PerHeadNorm(WeightInfo w, string name, int heads)
        => HfTransforms.ToF32(w, name, new Shape(heads, w.Shape.Elements() / heads));
}
