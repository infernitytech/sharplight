using System.Text.Json;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>mellum</c> (JetBrains Mellum 2): the Qwen3-MoE layer (per-head q/k norms, softmax router renormalized when
/// <c>norm_topk_prob</c>) with sliding-window layers from <c>layer_types</c> that rotate with their own base (<c>rope_parameters</c> per layer type)
/// and dense layers from <c>mlp_layer_types</c>, which the weights tell apart. The optional multi-token-prediction block (<c>mtp.*</c>) is not loaded.
/// </summary>
internal sealed class MellumHfMapping : HfMoeFamilyMapping
{
    private const string FullAttention = "full_attention";
    private const string SlidingAttention = "sliding_attention";

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["mellum"];

    /// <inheritdoc/>
    protected override float RopeBase(HfConfig c) => ThetaOf(c, FullAttention, 500000.0);

    /// <inheritdoc/>
    protected override MoeRouting Routing(HfConfig c)
        => new(c.GetInt("num_local_experts") ?? c.Int("num_experts"), c.Int("num_experts_per_tok"), c.Int("moe_intermediate_size"), "softmax",
            c.GetBool("norm_topk_prob") ?? true);

    /// <inheritdoc/>
    protected override void AddFamilyHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        if (AddSlidingLayers(md, c) is not null)
        {
            md["decoder.rope.local_base"] = ThetaOf(c, SlidingAttention, 10000.0);
        }
    }

    /// <inheritdoc/>
    protected override bool IsIgnored(string name) => base.IsIgnored(name) || name.StartsWith("mtp.", StringComparison.Ordinal);

    private static float ThetaOf(HfConfig c, string layerType, double fallback)
    {
        var rope = c.RopeParameters is { } parameters && parameters.TryGetProperty(layerType, out var element) && element.ValueKind == JsonValueKind.Object
            ? element
            : (JsonElement?)null;
        if (rope is { } r && r.TryGetProperty("rope_type", out var type) && type.GetString() is { } kind && kind != "default")
        {
            throw new NotSupportedException($"mellum: rope_type '{kind}' for {layerType} layers is not supported.");
        }

        return (float)(HfMappingHelpers.OptionalNumber(rope, "rope_theta") ?? fallback);
    }
}
