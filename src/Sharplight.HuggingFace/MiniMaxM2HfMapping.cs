namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>minimax_m2</c> (MiniMax-M2 / M2.1): partial rotary embedding, q/k RMS norms over the whole projection (all heads),
/// and Mixtral-named experts (<c>block_sparse_moe.experts.N.w1/w2/w3</c>) behind a sigmoid router with selection bias
/// (<c>e_score_correction_bias</c>) whose weights are renormalized. Multi-token-prediction modules are not used.
/// </summary>
internal sealed class MiniMaxM2HfMapping : HfMoeFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["minimax_m2"];

    /// <inheritdoc/>
    protected override int FeedForwardSize(HfConfig c) => c.Int("intermediate_size", "mlp_intermediate_size");

    /// <inheritdoc/>
    protected override int RopeDimension(HfConfig c, int headDim) => c.GetInt("rotary_dim") ?? base.RopeDimension(c, headDim);

    /// <inheritdoc/>
    protected override MoeRouting Routing(HfConfig c)
        => new(c.Int("num_local_experts", "num_experts"), c.Int("num_experts_per_tok"), c.Int("intermediate_size"), "sigmoid", Normalize: true);
}
