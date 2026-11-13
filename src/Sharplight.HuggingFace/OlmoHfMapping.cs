namespace Sharplight.HuggingFace;

/// <summary>Hugging Face <c>olmo</c> (OLMo 1): non-parametric layer norm (the checkpoint has no norm tensors), optional q/k/v clamping (<c>clip_qkv</c>), SwiGLU, rotary halves.</summary>
internal sealed class OlmoHfMapping : HfDecoderFamilyMapping
{
    private const float LayerNormEpsilon = 1e-5f;

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["olmo"];

    /// <inheritdoc/>
    protected override bool LayerNorm => true;

    /// <inheritdoc/>
    protected override float NormEpsilon(HfConfig c) => LayerNormEpsilon;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        if (c.GetDouble("clip_qkv") is > 0 and var clip)
        {
            md["decoder.attention.clip_qkv"] = (float)clip;
        }
    }
}
