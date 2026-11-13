namespace Sharplight.Gguf;

/// <summary>llama.cpp <c>starcoder2</c>: grouped-query attention with biases, layer norm with bias, plain biased GELU MLP, sliding window on every layer.</summary>
internal sealed class Starcoder2GgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["starcoder2"];

    /// <inheritdoc/>
    protected override bool LayerNorm => true;

    /// <inheritdoc/>
    protected override bool Gated => false;

    /// <inheritdoc/>
    protected override string Activation => "gelu_tanh";

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        if (g.Int("attention.sliding_window", 0) is > 0 and var window)
        {
            md[GgufDecoderHyperParameters.Prefix + "attention.sliding_window"] = window;
        }
    }
}
