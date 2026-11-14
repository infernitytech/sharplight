namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>refact</c> (Refact-1.6B): a Llama layer (RMS norms, SwiGLU, one key/value head) with ALiBi instead of a rotary embedding. llama.cpp fixes the
/// ALiBi maximum bias at 8 for this architecture rather than reading it from the file.
/// </summary>
internal sealed class RefactGgufMapping : GgufDecoderFamilyMapping
{
    private const float AlibiMaxBias = 8f;

    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["refact"];

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        md[GgufDecoderHyperParameters.Prefix + "rope.layers"] = new int[TrunkLayerCount(g)];
        md[GgufDecoderHyperParameters.Prefix + "attention.alibi_max_bias"] = AlibiMaxBias;
    }
}
