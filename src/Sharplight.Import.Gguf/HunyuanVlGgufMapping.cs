namespace Sharplight.Import.Gguf;

/// <summary>
/// llama.cpp <c>hunyuan_vl</c> (HunyuanOCR, Hunyuan-VL language model): the <c>hunyuan-dense</c> layer (q/k norms after rope, NTK alpha from
/// <c>rope.scaling.alpha</c> folded into the base) with XD-RoPE from <c>rope.dimension_sections</c>. The sections partition the rotated
/// elements of a head, so the two halves of a rotation pair can follow different position axes (<c>rope.sections_per_element</c>), as in the
/// reference implementation; text tokens carry their position on every axis, so text-only generation is plain rope.
/// </summary>
internal sealed class HunyuanVlGgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["hunyuan_vl"];

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        HunyuanDenseGgufMapping.AddHunyuanAttention(md, g, HeadDimOf(g));
        if (g.Raw("rope.dimension_sections") is not Array sections)
        {
            return;
        }

        var axes = new int[4];
        for (int i = 0; i < Math.Min(axes.Length, sections.Length); i++)
        {
            axes[i] = Convert.ToInt32(sections.GetValue(i));
        }

        string prefix = GgufDecoderHyperParameters.Prefix;
        md[prefix + "rope.sections"] = axes;
        md[prefix + "rope.sections_interleaved"] = false;
        md[prefix + "rope.sections_per_element"] = true;
    }
}
