namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>afmoe</c> (Arcee Trinity): sandwich norms around both sub-layers, per-head q/k norms, a sigmoid attention output gate and
/// token embeddings scaled by the square root of the hidden size. Every fourth layer is global (unless the file lists the pattern); only
/// the sliding-window layers are rotary. Leading dense layers, then a sigmoid router with selection bias and shared experts.
/// </summary>
internal sealed class AfmoeGgufMapping : GgufDecoderFamilyMapping
{
    private const int DefaultPattern = 4;

    private static readonly Dictionary<string, string> s_tensors = new(StringComparer.Ordinal) { ["attn_gate.weight"] = "attention.gate.weight" };

    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["afmoe"];

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string>? LayerTensorOverrides => s_tensors;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        string prefix = GgufDecoderHyperParameters.Prefix;
        var sliding = GgufMappingHelpers.SlidingLayers(g, TrunkLayerCount(g), DefaultPattern, denseFirst: false);
        if (g.Int("attention.sliding_window", 0) > 0)
        {
            md[prefix + "attention.sliding_window"] = g.Int("attention.sliding_window");
            md[prefix + "attention.sliding_layers"] = sliding;
            md[prefix + "rope.local_base"] = g.Float("rope.freq_base_swa", g.Float("rope.freq_base", GgufMappingHelpers.DefaultRopeBase));
        }

        md[prefix + "rope.layers"] = sliding;
        if (g.Int("expert_gating_func", 0) == 0 && g.Int("expert_count", 0) > 0)
        {
            md[prefix + "moe.router_activation"] = "sigmoid";   // llama.cpp's default for this architecture
        }

        md[prefix + "embedding_scale"] = MathF.Sqrt(g.Int("embedding_length"));
    }
}
