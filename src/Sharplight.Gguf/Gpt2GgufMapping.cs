namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>gpt2</c> and <c>starcoder</c> (StarCoder 1, SantaCoder: one key/value head): learned absolute position embeddings (no rotary), fused biased q/k/v
/// (split by the head counts), layer norm with bias, plain GELU MLP.
/// </summary>
internal sealed class Gpt2GgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["gpt2", "starcoder"];

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
    }
}
