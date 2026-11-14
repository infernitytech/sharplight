namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>llama-embed</c> (Llama-Embed-Nemotron): Llama layers run as an embedding model. As in llama.cpp the attention is causal unless the
/// file declares <c>attention.causal</c> = false; q/k rows are stored permuted for the interleaved rope like <c>llama</c>.
/// </summary>
internal sealed class LlamaEmbedGgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["llama-embed"];

    /// <inheritdoc/>
    protected override bool PermutesQk => true;
}
