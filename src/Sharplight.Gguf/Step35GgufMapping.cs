namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>step35</c> (Step-3.5): per-layer head counts, sliding-window layers from the per-layer pattern with their own rotary base,
/// the full-attention layers rotating half the rotary dimensions (and alone using the rope factors), per-head q/k norms and a per-head sigmoid
/// output gate (<c>attn_gate</c>, [heads, hidden]). Leading dense SwiGLU layers, then sigmoid-routed experts (llama.cpp's default when the file
/// names no gating function) with a selection bias plus a shared expert; the experts (<c>swiglu_clamp_exp</c>) and the dense / shared MLPs
/// (<c>swiglu_clamp_shexp</c>) clamp silu(gate) from above and the up projection on both sides, per layer.
/// </summary>
internal sealed class Step35GgufMapping : GgufDecoderFamilyMapping
{
    private static readonly Dictionary<string, string> s_tensors = new(StringComparer.Ordinal) { ["attn_gate.weight"] = "attention.gate.weight" };

    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["step35"];

    /// <inheritdoc/>
    protected override bool NormalizesExpertWeights => false;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string>? LayerTensorOverrides => s_tensors;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        string prefix = GgufDecoderHyperParameters.Prefix;
        int layerCount = TrunkLayerCount(g);
        md[prefix + "attention.layer_head_counts"] = GgufMappingHelpers.PerLayer(g, "attention.head_count", layerCount);
        md[prefix + "attention.layer_kv_head_counts"] = GgufMappingHelpers.PerLayer(g, "attention.head_count_kv", layerCount);
        md[prefix + "attention.sliding_window"] = g.Int("attention.sliding_window");
        md[prefix + "attention.sliding_layers"] = GgufMappingHelpers.SlidingLayers(g, layerCount, 0, denseFirst: false);
        int ropeDims = g.Int("rope.dimension_count", HeadDimOf(g));
        md[prefix + "rope.dim"] = ropeDims / 2;
        md[prefix + "rope.local_dim"] = g.Int("rope.dimension_count_swa", ropeDims);
        md[prefix + "rope.local_base"] = g.Float("rope.freq_base_swa", g.Float("rope.freq_base", GgufMappingHelpers.DefaultRopeBase));
        if (!md.ContainsKey(prefix + "moe.router_activation"))
        {
            md[prefix + "moe.router_activation"] = "sigmoid";
        }

        md[prefix + "moe.swiglu_clamp"] = GgufMappingHelpers.PerLayerFloat(g, "swiglu_clamp_exp", layerCount);
        md[prefix + "mlp.swiglu_clamp"] = GgufMappingHelpers.PerLayerFloat(g, "swiglu_clamp_shexp", layerCount);
    }

    /// <inheritdoc/>
    protected override (int Heads, int KvHeads) HeadsOf(GgufMetadataView g, int layer)
    {
        int layerCount = g.Int("block_count");
        return (GgufMappingHelpers.PerLayer(g, "attention.head_count", layerCount)[layer], GgufMappingHelpers.PerLayer(g, "attention.head_count_kv", layerCount)[layer]);
    }
}
