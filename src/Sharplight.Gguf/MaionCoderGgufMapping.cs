namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>maion-coder</c>: sliding-window layers from the per-layer pattern with rotary embedding (the full-attention layers are NoPE),
/// per-head q/k norms, sandwich norms around both sub-layers, and every <c>interleave_moe_layer_step</c>-th layer a softmax-routed SwiGLU
/// expert block with renormalized weights that runs in a latent space (<c>ffn_latent_down</c> / <c>ffn_latent_up</c>, size
/// <c>moe_latent_size</c>) while the router reads the full rows.
/// </summary>
internal sealed class MaionCoderGgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["maion-coder"];

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        string prefix = GgufDecoderHyperParameters.Prefix;
        int layerCount = g.Int("block_count");
        int[] sliding = GgufMappingHelpers.SlidingLayers(g, layerCount, 0, denseFirst: false);
        md[prefix + "attention.sliding_window"] = g.Int("attention.sliding_window");
        md[prefix + "attention.sliding_layers"] = sliding;
        md[prefix + "rope.layers"] = sliding;

        // The routing is fixed: softmax over all experts, top-k weights renormalized, unscaled.
        md[prefix + "moe.normalize_weights"] = true;
        md.Remove(prefix + "moe.weights_scale");
        md.Remove(prefix + "moe.router_activation");
    }
}
