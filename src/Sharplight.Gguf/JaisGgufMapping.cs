namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>jais</c> (Jais 13B/30B): layer norm with bias, a fused biased q/k/v projection, a biased SwiGLU MLP and ALiBi (when the file states
/// <c>attention.max_alibi_bias</c>) instead of a rotary embedding. Attention scores are scaled by 1/head_dim (muP); the converter has folded the embedding
/// and output multipliers into the weights.
/// </summary>
internal sealed class JaisGgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["jais"];

    /// <inheritdoc/>
    protected override bool LayerNorm => true;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        string prefix = GgufDecoderHyperParameters.Prefix;
        md[prefix + "rope.layers"] = new int[TrunkLayerCount(g)];
        md[prefix + "attention.scale"] = 1f / HeadDimOf(g);
        if (g.Float("attention.max_alibi_bias", 0f) is > 0f and var maxBias)
        {
            md[prefix + "attention.alibi_max_bias"] = maxBias;
        }
    }
}
