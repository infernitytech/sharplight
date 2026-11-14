namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>phimoe</c> (Phi-3.5-MoE, Phi-tiny-MoE): biased layer norms (epsilon <c>rms_norm_eps</c>), biased attention with an optional
/// sliding window and LongRoPE factors, top-2 SwiGLU experts (<c>block_sparse_moe.experts.N.w1/w3/w2</c>) routed by sparsemixer with jitter
/// ε = <c>router_jitter_noise</c>, and an optionally biased output head (<c>lm_head_bias</c>).
/// </summary>
internal sealed class PhimoeHfMapping : HfMoeFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["phimoe"];

    /// <inheritdoc/>
    protected override bool LayerNorm => true;

    /// <inheritdoc/>
    protected override MoeRouting Routing(HfConfig c)
        => new(c.Int("num_local_experts"), c.Int("num_experts_per_tok"), c.Int("intermediate_size"), "sparsemixer", Normalize: false);

    /// <inheritdoc/>
    protected override void AddFamilyHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        md["decoder.moe.sparsemixer_jitter"] = (float)(c.GetDouble("router_jitter_noise") ?? 0.01);
        if (c.GetInt("sliding_window") is > 0 and var window)
        {
            md["decoder.attention.sliding_window"] = window;
        }
    }
}
