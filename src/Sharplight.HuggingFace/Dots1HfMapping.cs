namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>dots1</c> (dots.llm1): the Qwen3 layer (per-head q/k norms) with leading dense layers and a DeepSeek-V3-style
/// router: sigmoid scores with selection bias (<c>e_score_correction_bias</c>), group-limited top-k, renormalized and scaled weights
/// plus shared experts.
/// </summary>
internal sealed class Dots1HfMapping : HfMoeFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["dots1"];

    /// <inheritdoc/>
    protected override MoeRouting Routing(HfConfig c)
        => new(c.Int("n_routed_experts"), c.Int("num_experts_per_tok"), c.Int("moe_intermediate_size"), "sigmoid",
            c.GetBool("norm_topk_prob") ?? false, (float)(c.GetDouble("routed_scaling_factor") ?? 1.0),
            c.GetInt("n_group") ?? 1, c.GetInt("topk_group") ?? 1);

    /// <inheritdoc/>
    protected override void AddFamilyHyperParameters(Dictionary<string, object> md, HfConfig c) => AddSlidingLayers(md, c);
}
