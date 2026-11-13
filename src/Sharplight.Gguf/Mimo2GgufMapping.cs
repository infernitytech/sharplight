namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>mimo2</c> (MiMo-V2-Flash): interleaved sliding-window layers (the per-layer pattern is stored) with learned attention sinks
/// and their own key/value head counts, a value head size smaller than the key head size, a constant factor on the attention output, a
/// separate rope base for the sliding layers, and a leading dense layer followed by sigmoid-routed experts with selection bias and no
/// shared expert. The appended NextN (multi-token-prediction) blocks are not used.
/// </summary>
internal sealed class Mimo2GgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["mimo2"];

    /// <inheritdoc/>
    protected override (int Heads, int KvHeads) HeadsOf(GgufMetadataView g, int layer)
        => (g.Int("attention.head_count"), KvHeadCounts(g, TrunkLayerCount(g))[layer]);

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        string prefix = GgufDecoderHyperParameters.Prefix;
        int layers = TrunkLayerCount(g);
        md[prefix + "attention.sliding_window"] = g.Int("attention.sliding_window");
        md[prefix + "attention.sliding_layers"] = GgufMappingHelpers.SlidingLayers(g, layers, defaultPattern: 0, denseFirst: false);
        md[prefix + "attention.layer_kv_head_counts"] = KvHeadCounts(g, layers);
        md[prefix + "attention.value_dim"] = g.Int("attention.value_length");
        md[prefix + "rope.local_base"] = g.Float("rope.freq_base_swa", g.Float("rope.freq_base", GgufMappingHelpers.DefaultRopeBase));
        if (g.Float("attention.value_scale", 1f) is var valueScale && valueScale != 1f)
        {
            md[prefix + "attention.value_scale"] = valueScale;
        }

        if (g.Int("expert_count", 0) > 0)
        {
            md[prefix + "moe.router_activation"] = "sigmoid";   // llama.cpp hard-codes the gating function and the renormalization
            md[prefix + "moe.normalize_weights"] = true;
        }
    }

    private static int[] KvHeadCounts(GgufMetadataView g, int layers)
    {
        if (g.Raw("attention.head_count_kv") is Array values)
        {
            return values.Cast<object>().Take(layers).Select(Convert.ToInt32).ToArray();
        }

        return Enumerable.Repeat(g.Int("attention.head_count_kv", g.Int("attention.head_count")), layers).ToArray();
    }
}
