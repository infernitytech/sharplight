using System.Text.Json;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>laguna</c> (poolside Laguna): per-head q/k norms, a softplus attention output gate (one scalar per head, or per channel
/// when the gate projection is that wide), a query head count that may differ between layers, sliding-window layers (<c>layer_types</c>) with
/// their own rope base and rotary width next to full-attention layers whose rope may be YaRN-scaled (<c>rope_parameters</c> per layer type),
/// and a leading dense layer followed by sigmoid-routed experts with selection bias (renormalized, scaled by <c>moe_routed_scaling_factor</c>)
/// plus one shared expert of its own width.
/// </summary>
internal sealed class LagunaHfMapping : HfMoeFamilyMapping
{
    private const string FullAttention = "full_attention";
    private const string SlidingAttention = "sliding_attention";
    private const string RouterBiasInExperts = "mlp.experts.e_score_correction_bias";

    private static readonly IReadOnlyDictionary<string, string> s_layerTensors = With(LlamaLayerTensors,
        ("self_attn.g_proj.weight", "attention.gate.weight"));

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["laguna"];

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> LayerTensors => s_layerTensors;

    /// <inheritdoc/>
    protected override int HeadCount(HfConfig c) => LayerHeadCounts(c).Max();

    /// <inheritdoc/>
    protected override int RopeDimension(HfConfig c, int headDim) => (int)(headDim * PartialRotaryFactor(c, FullAttention));

    /// <inheritdoc/>
    protected override float RopeBase(HfConfig c) => ThetaOf(c, FullAttention);

    /// <inheritdoc/>
    protected override MoeRouting Routing(HfConfig c)
    {
        if (c.GetDouble("moe_router_logit_softcapping") is > 0)
        {
            throw new NotSupportedException("laguna: moe_router_logit_softcapping is not supported.");
        }

        return new(c.Int("num_experts"), c.Int("num_experts_per_tok"), c.Int("moe_intermediate_size"), "sigmoid",
            Normalize: true, Scale: (float)(c.GetDouble("moe_routed_scaling_factor") ?? 1.0));
    }

    /// <inheritdoc/>
    protected override void AddFamilyHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        int headDim = HeadDimOf(c);
        md["decoder.attention.gate_activation"] = "softplus";
        md["decoder.attention.layer_head_counts"] = LayerHeadCounts(c);
        var layerTypes = c.GetStringArray("layer_types");
        if (layerTypes is not null && c.GetInt("sliding_window") is > 0 and var window && layerTypes.Contains(SlidingAttention))
        {
            md["decoder.attention.sliding_window"] = window;
            md["decoder.attention.sliding_layers"] = HfMappingHelpers.SlidingFlags(layerTypes);
            md["decoder.rope.local_base"] = ThetaOf(c, SlidingAttention);
            md["decoder.rope.local_dim"] = (int)(headDim * PartialRotaryFactor(c, SlidingAttention));
        }
    }

    /// <inheritdoc/>
    protected override void AddRopeScaling(Dictionary<string, object> md, HfConfig c)
    {
        if (FullScaling(c) is { } scaling && scaling.AttentionScale != 1f)
        {
            md["decoder.rope.attention_scale"] = scaling.AttentionScale;
        }
    }

    /// <inheritdoc/>
    protected override void AddRopeFactors(HfConfig c, HfWeightCollector collector)
    {
        if (FullScaling(c) is { } scaling)
        {
            collector.Add(HfMappingHelpers.F32Weight("rope.freq_factors", scaling.Factors));
        }
    }

    /// <inheritdoc/>
    protected override bool TryMapFamilyTensor(HfConfig c, string layerPrefix, string suffix, WeightInfo w, HfWeightCollector collector)
    {
        if (suffix != RouterBiasInExperts)
        {
            return false;
        }

        collector.Add(HfTransforms.ToF32(w, layerPrefix + "mlp.router_bias", new Shape(w.Shape.Elements())));
        return true;
    }

    private RopeScaling.Result? FullScaling(HfConfig c) => DecoderHfMapping.ComputeScaling(c, RopeDimension(c, HeadDimOf(c)), RopeBase(c));

    private int[] LayerHeadCounts(HfConfig c)
        => IntArray(c, "num_attention_heads_per_layer") ?? Enumerable.Repeat(c.Int("num_attention_heads"), LayerCount(c)).ToArray();

    private static JsonElement? RopeOf(HfConfig c, string layerType)
        => c.RopeParameters is { } parameters && parameters.TryGetProperty(layerType, out var element) && element.ValueKind == JsonValueKind.Object ? element : null;

    private static double PartialRotaryFactor(HfConfig c, string layerType)
        => HfMappingHelpers.OptionalNumber(RopeOf(c, layerType), "partial_rotary_factor") ?? c.RopeDouble("partial_rotary_factor", "partial_rotary_factor") ?? 1.0;

    private static float ThetaOf(HfConfig c, string layerType)
        => (float)(HfMappingHelpers.OptionalNumber(RopeOf(c, layerType), "rope_theta") ?? c.RopeDouble("rope_theta", "rope_theta") ?? 10000.0);
}
