namespace Sharplight.Gguf;

/// <summary>llama.cpp <c>nemotron</c> (Nemotron-4, Minitron, Nemotron-Mini): layer norm with bias (the converter folded the <c>1 + w</c> convention), squared-ReLU MLP without gate, partial rotary.</summary>
internal sealed class NemotronGgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["nemotron"];

    /// <inheritdoc/>
    protected override bool LayerNorm => true;

    /// <inheritdoc/>
    protected override bool Gated => false;

    /// <inheritdoc/>
    protected override string Activation => "relu2";
}
