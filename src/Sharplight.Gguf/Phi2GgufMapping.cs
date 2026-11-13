namespace Sharplight.Gguf;

/// <summary>llama.cpp <c>phi2</c> (Phi-1/1.5/2): layer norm with bias, parallel residual around one shared norm, partial rotary, plain biased GELU MLP, biased output head.</summary>
internal sealed class Phi2GgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["phi2"];

    /// <inheritdoc/>
    protected override bool LayerNorm => true;

    /// <inheritdoc/>
    protected override bool Gated => false;

    /// <inheritdoc/>
    protected override string Activation => "gelu_tanh";

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        md[GgufDecoderHyperParameters.Prefix + "residual.parallel"] = true;
    }
}
