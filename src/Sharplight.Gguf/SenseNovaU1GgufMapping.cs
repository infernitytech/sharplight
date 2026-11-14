namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>sensenova_u1</c> (SenseNova-U1, understanding branch): Qwen3 whose q/k norm weight covers a temporal and a spatial half of each
/// head that are normalized separately (canonical <c>attention.qk_norm_groups</c> = 2). M-RoPE rotates the temporal half by the token position
/// and the spatial quarters by the image row and column (<c>rope.freq_base_spatial</c>); text tokens sit at row and column 0, so for text the
/// rotation is a NeoX rope over the first half of the head.
/// </summary>
internal sealed class SenseNovaU1GgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["sensenova_u1"];

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        string prefix = GgufDecoderHyperParameters.Prefix;
        md[prefix + "rope.dim"] = HeadDimOf(g) / 2;
        md[prefix + "attention.qk_norm_groups"] = 2;
    }
}
