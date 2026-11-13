namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>bloom</c> (BLOOM): ALiBi position biases (maximum bias 8, fixed by llama.cpp) instead of a rotary embedding, layer norm with bias also on the
/// embeddings (<c>token_embd_norm</c>), fused biased q/k/v and a plain GELU MLP.
/// </summary>
internal sealed class BloomGgufMapping : GgufDecoderFamilyMapping
{
    private const float AlibiMaxBias = 8f;

    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["bloom"];

    /// <inheritdoc/>
    protected override bool LayerNorm => true;

    /// <inheritdoc/>
    protected override bool Gated => false;

    /// <inheritdoc/>
    protected override string Activation => "gelu_tanh";

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        md[GgufDecoderHyperParameters.Prefix + "rope.layers"] = new int[g.Int("block_count")];
        md[GgufDecoderHyperParameters.Prefix + "attention.alibi_max_bias"] = AlibiMaxBias;
    }
}
