namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>glm4</c> (GLM-4-0414 dense, GLM-Z1, and the GLM-4.1V language model) and <c>chatglm</c> (ChatGLM3, GLM-4-chat): partial rotary
/// embedding on interleaved pairs (stored unpermuted), fused gate+up projection (split on import), sandwich norms for GLM-4 (post-attention
/// and post-MLP output norms). GLM-4.1V files carry <c>rope.dimension_sections</c> (sectioned, non-interleaved M-RoPE); llama.cpp's
/// converter then reorders the rotated q/k rows to the neox layout (its M-RoPE kernel only rotates halves), so those files use the neox style.
/// </summary>
internal sealed class Glm4GgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["glm4", "chatglm"];

    /// <inheritdoc/>
    protected override string RopeStyle => "gptj";

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        if (g.Raw("rope.dimension_sections") is not Array sections)
        {
            return;
        }

        var axes = new int[4];
        for (int i = 0; i < Math.Min(axes.Length, sections.Length); i++)
        {
            axes[i] = Convert.ToInt32(sections.GetValue(i));
        }

        md[GgufDecoderHyperParameters.Prefix + "rope.sections"] = axes;
        md[GgufDecoderHyperParameters.Prefix + "rope.sections_interleaved"] = false;
        md[GgufDecoderHyperParameters.Prefix + "rope.style"] = "neox";
    }
}
