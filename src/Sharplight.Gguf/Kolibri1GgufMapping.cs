namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>kolibri1</c> (Kolibri-1): sliding-window layers from the per-layer pattern with the rotary embedding on those layers only
/// (the global layers are NoPE), per-head q/k norms, sandwich norms around both sub-layers, and routed SwiGLU experts plus a shared expert.
/// The experts are selected by the top-k of the router logits plus the selection bias (<c>exp_probs_b</c>), their weights are the router
/// activation of the unbiased logits.
/// </summary>
internal sealed class Kolibri1GgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["kolibri1"];

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        string prefix = GgufDecoderHyperParameters.Prefix;
        int[] sliding = GgufMappingHelpers.SlidingLayers(g, g.Int("block_count"), 0, denseFirst: false);
        md[prefix + "attention.sliding_window"] = g.Int("attention.sliding_window");
        md[prefix + "attention.sliding_layers"] = sliding;
        md[prefix + "rope.layers"] = sliding;
        md[prefix + "moe.selection_bias_on_logits"] = true;
    }
}
