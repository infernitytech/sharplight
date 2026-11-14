namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>eurobert</c> (EuroBERT): Llama layers (RMS norms, rotary embedding, SwiGLU) with bidirectional attention, an embedding model
/// (canonical <c>attention.causal</c> = false, whatever the file says).
/// </summary>
internal sealed class EuroBertGgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["eurobert"];

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
        => md[GgufDecoderHyperParameters.Prefix + "attention.causal"] = false;
}
