namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Clover (<c>clover_moe</c>, remote code): Qwen3.5-MoE layers decoded causally, with plain 1-D rope. Its block-diffusion preview tensors (the
/// right-hand convolution of the linear-attention layers and the block-softmax memory parameters) are only used when it previews a block of
/// masked positions; causal decoding (the commit pass, and llama.cpp's <c>clover_moe</c>) does not read them.
/// </summary>
internal sealed partial class Qwen35HfMapping
{
    private const string CloverModelType = "clover_moe";

    /// <summary>Linear-attention tensors of Clover's block-diffusion preview.</summary>
    private static readonly string[] s_cloverPreviewSuffixes =
        [".linear_attn.conv_right_weight", ".linear_attn.bd_softmax_logit_scale", ".linear_attn.bd_softmax_history_bias", ".linear_attn.bd_softmax_history_c"];

    private static bool IsClover(HfConfig c) => c.ModelType == CloverModelType;

    /// <summary>Whether <paramref name="name"/> is one of Clover's preview tensors, which causal decoding leaves out.</summary>
    private static bool IsCloverPreviewTensor(HfConfig c, string name)
        => IsClover(c) && s_cloverPreviewSuffixes.Any(suffix => name.EndsWith(suffix, StringComparison.Ordinal));
}
