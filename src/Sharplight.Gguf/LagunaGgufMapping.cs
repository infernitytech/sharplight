namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>laguna</c> (poolside Laguna): per-head q/k norms, a softplus attention output gate (per head, or per channel when the gate
/// tensor is that wide), a query head count that may differ between layers, optional sliding-window layers (the first layer of every four is
/// global) with their own rope base and rotary width next to full-attention layers whose rope may be YaRN-scaled, and a leading dense layer
/// followed by sigmoid-routed experts with selection bias plus one shared expert.
/// </summary>
internal sealed class LagunaGgufMapping : GgufDecoderFamilyMapping
{
    private const int DefaultPattern = 4;

    private static readonly Dictionary<string, string> s_tensors = new(StringComparer.Ordinal) { ["attn_gate.weight"] = "attention.gate.weight" };

    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["laguna"];

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string>? LayerTensorOverrides => s_tensors;

    /// <inheritdoc/>
    protected override (int Heads, int KvHeads) HeadsOf(GgufMetadataView g, int layer)
        => (HeadCounts(g)[layer], g.Int("attention.head_count_kv"));

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        string prefix = GgufDecoderHyperParameters.Prefix;
        int layers = TrunkLayerCount(g);
        md[prefix + "attention.gate_activation"] = "softplus";
        md[prefix + "attention.layer_head_counts"] = HeadCounts(g);
        if (g.Int("attention.sliding_window", 0) > 0)
        {
            md[prefix + "attention.sliding_window"] = g.Int("attention.sliding_window");
            md[prefix + "attention.sliding_layers"] = GgufMappingHelpers.SlidingLayers(g, layers, DefaultPattern, denseFirst: true);
            md[prefix + "rope.local_base"] = g.Float("rope.freq_base_swa", g.Float("rope.freq_base", GgufMappingHelpers.DefaultRopeBase));
            md[prefix + "rope.local_dim"] = g.Int("rope.dimension_count_swa", HeadDimOf(g));
        }

        if (g.Int("expert_gating_func", 0) == 0 && g.Int("expert_count", 0) > 0)
        {
            md[prefix + "moe.router_activation"] = "sigmoid";   // llama.cpp's default for this architecture
        }
    }

    private static int[] HeadCounts(GgufMetadataView g)
    {
        int layers = g.Int("block_count");
        if (g.Raw("attention.head_count") is Array values)
        {
            return values.Cast<object>().Take(layers).Select(Convert.ToInt32).ToArray();
        }

        return Enumerable.Repeat(g.Int("attention.head_count"), layers).ToArray();
    }
}
