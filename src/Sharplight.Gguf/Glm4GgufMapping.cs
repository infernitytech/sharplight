namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>glm4</c> (GLM-4-0414 dense, GLM-Z1) and <c>chatglm</c> (ChatGLM3, GLM-4-chat): partial rotary embedding on interleaved
/// pairs (stored unpermuted), fused gate+up projection (split on import), sandwich norms for GLM-4 (post-attention and post-MLP output norms).
/// </summary>
internal sealed class Glm4GgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["glm4", "chatglm"];

    /// <inheritdoc/>
    protected override string RopeStyle => "gptj";
}
