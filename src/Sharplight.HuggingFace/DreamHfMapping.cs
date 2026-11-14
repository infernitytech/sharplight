using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>Dream</c> (Dream 7B, remote code <c>DreamModel</c>): Qwen2 layers with bidirectional attention, a masked-diffusion language
/// model whose logits row i predicts position i + 1 (canonical <c>attention.causal</c> = false, <c>diffusion.shift_logits</c> = true, the
/// config's <c>mask_token_id</c> as the tokenizer's mask token).
/// </summary>
internal sealed class DreamHfMapping : HfDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["Dream"];

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, HfConfig c) => AddDiffusion(md, c, shiftLogits: true);

    /// <summary>The bidirectional-attention, logit-shift and mask-token keys of a masked-diffusion model.</summary>
    internal static void AddDiffusion(Dictionary<string, object> md, HfConfig c, bool shiftLogits)
    {
        md["decoder.attention.causal"] = false;
        md["decoder.diffusion.shift_logits"] = shiftLogits;
        if (c.GetInt("mask_token_id") is { } mask)
        {
            md[CanonicalKeys.TokenizerMaskId] = mask;
        }
    }
}
