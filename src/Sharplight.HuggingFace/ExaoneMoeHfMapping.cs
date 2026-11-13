namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>exaone_moe</c> (K-EXAONE): the pre-norm layer with per-head q/k norms, interleaved sliding-window layers from
/// <c>layer_types</c> that are the only ones with a rotary embedding, leading dense layers and a sigmoid router with selection bias
/// (<c>e_score_correction_bias</c>), optional group-limited top-k, renormalized and scaled weights plus shared experts.
/// </summary>
internal sealed class ExaoneMoeHfMapping : HfMoeFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["exaone_moe"];

    /// <inheritdoc/>
    protected override MoeRouting Routing(HfConfig c)
        => new(c.Int("num_experts"), c.Int("num_experts_per_tok"), c.Int("moe_intermediate_size"), "sigmoid",
            c.GetBool("norm_topk_prob") ?? true, (float)(c.GetDouble("routed_scaling_factor") ?? 1.0),
            c.GetInt("n_group") ?? 1, c.GetInt("topk_group") ?? 1);

    /// <inheritdoc/>
    protected override void AddFamilyHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        if (AddSlidingLayers(md, c) is { } sliding)
        {
            md["decoder.rope.layers"] = sliding;
        }
    }
}
