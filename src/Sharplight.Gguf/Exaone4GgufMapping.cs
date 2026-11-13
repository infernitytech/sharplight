namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>exaone4</c> (EXAONE 4.0): norms only after the attention and MLP outputs, per-head q/k norms and, for the 64-layer model,
/// sliding-window layers (every fourth layer is global) that are the only ones with a rotary embedding.
/// </summary>
internal sealed class Exaone4GgufMapping : GgufDecoderFamilyMapping
{
    private const int SlidingModelLayerCount = 64;
    private const int SlidingPattern = 4;
    private const int DefaultWindow = 4096;

    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["exaone4"];

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        string prefix = GgufDecoderHyperParameters.Prefix;
        md[prefix + "pre_norm"] = false;
        int layerCount = g.Int("block_count");
        if (layerCount != SlidingModelLayerCount)
        {
            return;
        }

        var sliding = GgufMappingHelpers.SlidingLayers(layerCount, SlidingPattern);
        md[prefix + "attention.sliding_window"] = g.Int("attention.sliding_window", DefaultWindow);
        md[prefix + "attention.sliding_layers"] = sliding;
        md[prefix + "rope.layers"] = sliding;
        md[prefix + "rope.local_base"] = g.Float("rope.freq_base_swa", g.Float("rope.freq_base", GgufMappingHelpers.DefaultRopeBase));
    }
}
