namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>hunyuan-dense</c> (Hunyuan 0.5B-7B): per-head q/k RMS norms applied after the rotary embedding, and NTK-aware rope
/// scaling folded into the base (<c>base · alpha^(dim / (dim − 2))</c>).
/// </summary>
internal sealed class HunyuanDenseGgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["hunyuan-dense"];

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        string prefix = GgufDecoderHyperParameters.Prefix;
        md[prefix + "attention.qk_norm_after_rope"] = true;
        float alpha = g.Float("rope.scaling.alpha", 0f);
        if (alpha > 0f)
        {
            int headDim = HeadDimOf(g);
            md[prefix + "rope.base"] = g.Float("rope.freq_base", GgufMappingHelpers.DefaultRopeBase) * MathF.Pow(alpha, headDim / (float)(headDim - 2));
        }
    }
}
