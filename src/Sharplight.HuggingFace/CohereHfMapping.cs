namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>cohere</c> (Command-R, Aya) and <c>cohere2</c> (Command-R7B, Command-A): layer norm without bias, attention and MLP in parallel around one shared norm,
/// interleaved rotary pairs, logits scaled by <c>logit_scale</c>. Cohere2 alternates sliding-window layers (with rotary embedding) and global layers without it.
/// Per-head q/k layer norms (<c>use_qk_norm</c>) are not supported.
/// </summary>
internal sealed class CohereHfMapping : HfDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["cohere", "cohere2"];

    /// <inheritdoc/>
    protected override bool LayerNorm => true;

    /// <inheritdoc/>
    protected override string RopeStyle => "gptj";

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        if (c.GetBool("use_qk_norm") == true)
        {
            throw new NotSupportedException($"{c.ModelType}: per-head q/k layer norms (use_qk_norm) are not supported.");
        }

        md["decoder.residual.parallel"] = true;
        md["decoder.logit_scale"] = (float)(c.GetDouble("logit_scale") ?? 1.0);
        if (c.ModelType == "cohere2" && AddSlidingLayers(md, c) is { } sliding)
        {
            md["decoder.rope.layers"] = sliding;
        }
    }
}
