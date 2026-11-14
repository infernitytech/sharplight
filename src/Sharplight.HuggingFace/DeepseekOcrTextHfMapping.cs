namespace Sharplight.HuggingFace;

/// <summary>
/// The language model of DeepSeek-OCR, DeepSeek-OCR-2 and Unlimited-OCR (llama.cpp <c>deepseek2-ocr</c>): DeepSeek-V2's layer without latent
/// attention, i.e. Llama-named multi-head attention with rotary over the whole head (the configs' <c>qk_rope_head_dim</c> / <c>v_head_dim</c>
/// describe no latent attention and are ignored), leading dense SwiGLU layers (<c>first_k_dense_replace</c> / <c>mlp_layer_types</c>; the weights
/// tell them apart), then a softmax router over SwiGLU experts (top-k weights renormalized only with <c>norm_topk_prob</c>, scaled by
/// <c>routed_scaling_factor</c>) plus always-on shared experts. Used through <see cref="DeepseekOcrHfMapping"/>, which strips the tower.
/// </summary>
internal sealed class DeepseekOcrTextHfMapping : HfMoeFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["deepseek_ocr2_text", "unlimited_ocr_text"];

    /// <inheritdoc/>
    protected override int RopeDimension(HfConfig c, int headDim) => headDim;

    /// <inheritdoc/>
    protected override float NormEpsilon(HfConfig c) => (float)(c.GetDouble("rms_norm_eps") ?? 1e-6);

    /// <inheritdoc/>
    protected override MoeRouting? Routing(HfConfig c)
    {
        int experts = c.GetInt("n_routed_experts", "num_experts") ?? 0;
        if (experts <= 0)
        {
            return null;
        }

        string method = c.GetString("topk_method") ?? "greedy";
        int groups = c.GetInt("n_group") ?? 1;
        if (method != "greedy" && groups > 1)
        {
            throw new NotSupportedException($"DeepSeek-OCR: topk_method '{method}' with {groups} expert groups is not supported.");
        }

        return new MoeRouting(experts, c.Int("num_experts_per_tok"), c.Int("moe_intermediate_size"), "softmax",
            c.GetBool("norm_topk_prob") ?? false, (float)(c.GetDouble("routed_scaling_factor") ?? 1.0));
    }
}
