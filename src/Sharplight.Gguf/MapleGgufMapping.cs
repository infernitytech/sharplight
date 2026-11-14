namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>maple</c>: sliding-window layers from the per-layer pattern, rotary embedding on those layers only (the global layers are
/// NoPE) with their own base (<c>rope.freq_base_swa</c>), per-head q/k norms, and softmax-routed SwiGLU experts with renormalized weights
/// whose gate projection is capped and up projection clamped per layer (<c>swiglu_clamp_exp</c>, llama.cpp <c>ggml_swiglu_clamp</c>).
/// </summary>
internal sealed class MapleGgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["maple"];

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        string prefix = GgufDecoderHyperParameters.Prefix;
        int layerCount = g.Int("block_count");
        int[] sliding = GgufMappingHelpers.SlidingLayers(g, layerCount, 0, denseFirst: false);
        md[prefix + "attention.sliding_window"] = g.Int("attention.sliding_window");
        md[prefix + "attention.sliding_layers"] = sliding;
        md[prefix + "rope.layers"] = sliding;
        md[prefix + "rope.local_base"] = g.Float("rope.freq_base_swa", g.Float("rope.freq_base", GgufMappingHelpers.DefaultRopeBase));

        // The routing is fixed: softmax over all experts, top-k weights renormalized, unscaled.
        md[prefix + "moe.normalize_weights"] = true;
        md.Remove(prefix + "moe.weights_scale");
        md.Remove(prefix + "moe.router_activation");
        md[prefix + "moe.swiglu_clamp"] = GgufMappingHelpers.PerLayerFloat(g, "swiglu_clamp_exp", layerCount);
        md[prefix + "moe.swiglu_clamp_input"] = true;
    }
}
