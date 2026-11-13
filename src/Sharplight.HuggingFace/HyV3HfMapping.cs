namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>hy_v3</c> (Hunyuan 3): the Qwen3 layer (per-head q/k norms before rope) with a leading dense layer, then fused
/// experts behind a sigmoid router with selection bias (<c>e_score_correction_bias</c>) whose weights are always renormalized and
/// scaled by <c>router_scaling_factor</c>, plus shared experts.
/// </summary>
internal sealed class HyV3HfMapping : HfMoeFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["hy_v3"];

    /// <inheritdoc/>
    protected override MoeRouting Routing(HfConfig c)
        => new(c.Int("num_experts", "num_local_experts"), c.Int("num_experts_per_tok"), c.Int("moe_intermediate_size"), "sigmoid",
            Normalize: true, Scale: (float)(c.GetDouble("router_scaling_factor") ?? 1.0));
}
