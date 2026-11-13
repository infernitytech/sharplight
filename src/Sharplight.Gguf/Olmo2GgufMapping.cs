namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>olmo2</c> (OLMo 2, OLMo 3): norms only after the attention and MLP outputs (no norms in front), full-width q/k norms and,
/// for OLMo 3, sliding-window layers (every fourth layer is global) whose rope uses the local base without frequency scaling.
/// </summary>
internal sealed class Olmo2GgufMapping : GgufDecoderFamilyMapping
{
    private const int DefaultSlidingPattern = 4;

    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["olmo2"];

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        string prefix = GgufDecoderHyperParameters.Prefix;
        md[prefix + "pre_norm"] = false;
        int window = g.Int("attention.sliding_window", 0);
        if (window <= 0)
        {
            return;
        }

        int layerCount = g.Int("block_count");
        md[prefix + "attention.sliding_window"] = window;
        md[prefix + "attention.sliding_layers"] = g.Raw("attention.sliding_window_pattern") is Array
            ? PerLayerFlags(g, "attention.sliding_window_pattern", layerCount)
            : GgufMappingHelpers.SlidingLayers(layerCount, g.Int("attention.sliding_window_pattern", DefaultSlidingPattern));
        md[prefix + "rope.local_base"] = g.Float("rope.freq_base_swa", g.Float("rope.freq_base", GgufMappingHelpers.DefaultRopeBase));
    }

    private static int[] PerLayerFlags(GgufMetadataView g, string key, int layerCount)
    {
        var values = (Array)g.Raw(key)!;
        return Enumerable.Range(0, layerCount).Select(il => Convert.ToBoolean(values.GetValue(il)) ? 1 : 0).ToArray();
    }
}
