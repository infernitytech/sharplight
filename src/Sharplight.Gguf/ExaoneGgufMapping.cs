namespace Sharplight.Gguf;

/// <summary>llama.cpp <c>exaone</c> (EXAONE 3.0/3.5): the llama graph with rotary halves (no permutation); llama3 scaling arrives as <c>rope_freqs</c>.</summary>
internal sealed class ExaoneGgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["exaone"];
}
