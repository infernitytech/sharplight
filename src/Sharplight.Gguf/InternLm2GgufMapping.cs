namespace Sharplight.Gguf;

/// <summary>llama.cpp <c>internlm2</c> (InternLM2/2.5, split into separate q/k/v by the converter): the llama graph, rows permuted for interleaved rope.</summary>
internal sealed class InternLm2GgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["internlm2"];

    /// <inheritdoc/>
    protected override bool PermutesQk => true;
}
