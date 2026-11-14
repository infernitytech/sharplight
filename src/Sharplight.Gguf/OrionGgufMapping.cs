namespace Sharplight.Gguf;

/// <summary>llama.cpp <c>orion</c> (Orion-14B): layer norm with bias (although the source config names it RMS), neox rotary over the whole head, SwiGLU MLP.</summary>
internal sealed class OrionGgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["orion"];

    /// <inheritdoc/>
    protected override bool LayerNorm => true;
}
