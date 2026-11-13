namespace Sharplight.Gguf;

/// <summary>llama.cpp <c>gemma</c> (Gemma 1): multi-query attention, GeGLU, embeddings scaled by sqrt(hidden); the converter already folded the <c>1 + w</c> norm convention.</summary>
internal sealed class Gemma1GgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["gemma"];

    /// <inheritdoc/>
    protected override string Activation => "gelu_tanh";

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        md[GgufDecoderHyperParameters.Prefix + "embedding_scale"] = MathF.Sqrt(g.Int("embedding_length"));
    }
}
