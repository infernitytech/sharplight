namespace Sharplight.Import.HuggingFace;

/// <summary>Hugging Face <c>stablelm</c> (StableLM 2): layer norm with bias, partial rotary, SwiGLU, optional parallel residual (12B). Per-head q/k layer norms (<c>qk_layernorm</c>) are not supported.</summary>
internal sealed class StableLmHfMapping : HfDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["stablelm", "stablelm_epoch"];

    /// <inheritdoc/>
    protected override bool LayerNorm => true;

    /// <inheritdoc/>
    public override IEnumerable<string> Validate(HfConfig config)
    {
        if (config.GetBool("qk_layernorm") == true)
        {
            yield return "stablelm: per-head q/k layer norms (qk_layernorm) are not supported.";
        }
    }

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        md["decoder.residual.parallel"] = c.GetBool("use_parallel_residual") ?? false;
    }
}
