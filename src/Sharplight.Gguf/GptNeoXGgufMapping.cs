namespace Sharplight.Gguf;

/// <summary>llama.cpp <c>gptneox</c> (Pythia, StableLM-alpha, RedPajama): fused biased q/k/v, layer norm with bias, partial rotary, plain GELU MLP, optional parallel residual with two norms.</summary>
internal sealed class GptNeoXGgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["gptneox"];

    /// <inheritdoc/>
    protected override bool LayerNorm => true;

    /// <inheritdoc/>
    protected override bool Gated => false;

    /// <inheritdoc/>
    protected override string Activation => "gelu_tanh";

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        md[GgufDecoderHyperParameters.Prefix + "residual.parallel"] = g.Bool("use_parallel_residual", true);
    }
}
