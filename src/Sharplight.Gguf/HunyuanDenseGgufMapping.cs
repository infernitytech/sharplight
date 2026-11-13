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
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g) => AddHunyuanAttention(md, g, HeadDimOf(g));

    /// <summary>The Hunyuan attention conventions shared with <c>hunyuan-moe</c>: q/k norms after rope and the NTK-alpha rope base.</summary>
    internal static void AddHunyuanAttention(Dictionary<string, object> md, GgufMetadataView g, int headDim)
    {
        string prefix = GgufDecoderHyperParameters.Prefix;
        md[prefix + "attention.qk_norm_after_rope"] = true;
        float alpha = g.Float("rope.scaling.alpha", 0f);
        if (alpha > 0f)
        {
            md[prefix + "rope.base"] = g.Float("rope.freq_base", GgufMappingHelpers.DefaultRopeBase) * MathF.Pow(alpha, headDim / (float)(headDim - 2));
        }
    }
}
