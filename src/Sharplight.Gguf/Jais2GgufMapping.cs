namespace Sharplight.Gguf;

/// <summary>llama.cpp <c>jais2</c> (Jais 2): layer norms with bias, neox rotary over the whole head, a plain biased squared-ReLU MLP.</summary>
internal sealed class Jais2GgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["jais2"];

    /// <inheritdoc/>
    protected override bool LayerNorm => true;

    /// <inheritdoc/>
    protected override bool Gated => false;

    /// <inheritdoc/>
    protected override string Activation => "relu2";
}
