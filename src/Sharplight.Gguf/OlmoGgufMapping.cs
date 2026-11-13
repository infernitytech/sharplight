namespace Sharplight.Gguf;

/// <summary>llama.cpp <c>olmo</c> (OLMo 1): non-parametric layer norm (no norm tensors), optional q/k/v clamping, SwiGLU, rows permuted for interleaved rope.</summary>
internal sealed class OlmoGgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["olmo"];

    /// <inheritdoc/>
    protected override bool LayerNorm => true;

    /// <inheritdoc/>
    protected override bool PermutesQk => true;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        if (g.Has("attention.clamp_kqv"))
        {
            md[GgufDecoderHyperParameters.Prefix + "attention.clip_qkv"] = g.Float("attention.clamp_kqv");
        }
    }
}
