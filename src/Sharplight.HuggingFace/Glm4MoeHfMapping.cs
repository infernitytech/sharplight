namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>glm4_moe</c> (GLM-4.5 / 4.6 / 4.7): partial rotary embedding (rotate-half), optional per-head q/k norms, leading
/// dense layers (<c>first_k_dense_replace</c>) and a sigmoid router with selection bias (<c>e_score_correction_bias</c>), optional
/// group-limited top-k, renormalized and scaled weights plus shared experts. The trailing next-token-prediction layers are dropped.
/// </summary>
internal sealed class Glm4MoeHfMapping : HfMoeFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["glm4_moe"];

    /// <inheritdoc/>
    protected override MoeRouting Routing(HfConfig c)
        => new(c.Int("n_routed_experts", "num_local_experts"), c.Int("num_experts_per_tok"), c.Int("moe_intermediate_size"), "sigmoid",
            c.GetBool("norm_topk_prob") ?? true, (float)(c.GetDouble("routed_scaling_factor") ?? 1.0),
            c.GetInt("n_group") ?? 1, c.GetInt("topk_group") ?? 1);
}
