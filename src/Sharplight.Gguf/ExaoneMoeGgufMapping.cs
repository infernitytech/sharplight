namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>exaone-moe</c> (K-EXAONE): pre-norm layers with per-head q/k norms, sliding-window layers (every fourth layer is global,
/// unless the file lists the pattern) that are the only ones with a rotary embedding, and routed experts with a selection bias plus a
/// shared expert after the leading dense layers. The NextN (MTP) block is not used.
/// </summary>
internal sealed class ExaoneMoeGgufMapping : GgufDecoderFamilyMapping
{
    private const int DefaultPattern = 4;
    private const int DefaultWindow = 128;

    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["exaone-moe"];

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        string prefix = GgufDecoderHyperParameters.Prefix;
        var sliding = GgufMappingHelpers.SlidingLayers(g, TrunkLayerCount(g), DefaultPattern, denseFirst: false);
        md[prefix + "attention.sliding_window"] = g.Int("attention.sliding_window", DefaultWindow);
        md[prefix + "attention.sliding_layers"] = sliding;
        md[prefix + "rope.layers"] = sliding;
        md[prefix + "rope.local_base"] = g.Float("rope.freq_base_swa", g.Float("rope.freq_base", GgufMappingHelpers.DefaultRopeBase));
        md[prefix + "rope.local_freq_factors"] = true;
    }
}
