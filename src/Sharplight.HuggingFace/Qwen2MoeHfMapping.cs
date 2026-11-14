namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>qwen2_moe</c> (Qwen1.5-MoE-A2.7B, Qwen2-57B-A14B): the Qwen2 layer (q/k/v biases) with a softmax router over the experts
/// (renormalized when <c>norm_topk_prob</c>) plus a shared expert scaled per token by sigmoid(h · <c>shared_expert_gate</c>). Layers listed in
/// <c>mlp_only_layers</c> (or off the <c>decoder_sparse_step</c> grid) keep a dense MLP; the weights tell them apart.
/// </summary>
internal sealed class Qwen2MoeHfMapping : HfMoeFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["qwen2_moe"];

    /// <inheritdoc/>
    protected override MoeRouting Routing(HfConfig c)
        => new(c.Int("num_experts"), c.Int("num_experts_per_tok"), c.Int("moe_intermediate_size"), "softmax", c.GetBool("norm_topk_prob") ?? false);

    /// <inheritdoc/>
    protected override void AddFamilyHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        if (c.GetBool("use_sliding_window") == true)
        {
            throw new NotSupportedException("qwen2_moe: use_sliding_window is not supported.");
        }
    }
}
