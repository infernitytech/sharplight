namespace Sharplight.Gguf;

/// <summary>llama.cpp <c>xverse</c> (XVERSE): the llama graph, rows permuted for interleaved rope.</summary>
internal sealed class XverseGgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["xverse"];

    /// <inheritdoc/>
    protected override bool PermutesQk => true;
}
