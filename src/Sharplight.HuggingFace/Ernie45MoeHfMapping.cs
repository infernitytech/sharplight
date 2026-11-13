namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>ernie4_5_moe</c> (ERNIE-4.5-21B-A3B / 300B-A47B): the ERNIE 4.5 layer (interleaved rotary pairs) with a softmax
/// router whose top-k is chosen on the probabilities plus the selection bias (<c>moe_statics.e_score_correction_bias</c>), the
/// unbiased probabilities renormalized, and shared experts; layers outside the MoE range are dense.
/// </summary>
internal sealed class Ernie45MoeHfMapping : HfMoeFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["ernie4_5_moe"];

    /// <inheritdoc/>
    protected override string RopeStyle => "gptj";

    /// <inheritdoc/>
    protected override MoeRouting Routing(HfConfig c)
        => new(c.Int("moe_num_experts", "num_experts"), c.Int("moe_k", "num_experts_per_tok"), c.Int("moe_intermediate_size"), "softmax", Normalize: true);
}
