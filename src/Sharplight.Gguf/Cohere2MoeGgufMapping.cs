namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>cohere2moe</c> (Command A MoE): the Cohere2 layer (parallel residual around one norm, layer norm or RMS norm by the epsilon key the
/// file carries, interleaved rotary pairs stored unpermuted, logits scaled by <c>logit_scale</c>) with sliding-window layers from the per-layer
/// pattern, rotary embedding on the sliding and the leading dense layers only (all with the global base), and routed experts plus a shared expert
/// whose sum llama.cpp halves.
/// </summary>
internal sealed class Cohere2MoeGgufMapping : GgufDecoderFamilyMapping
{
    private const int DefaultPattern = 4;
    private const float AverageScale = 0.5f;

    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["cohere2moe"];

    /// <inheritdoc/>
    protected override string RopeStyle => "gptj";

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        string prefix = GgufDecoderHyperParameters.Prefix;
        if (!g.Has("attention.layer_norm_rms_epsilon"))
        {
            md[prefix + "norm.type"] = "layer";
        }

        md[prefix + "residual.parallel"] = true;
        md[prefix + "logit_scale"] = g.Float("logit_scale", 1f);
        int layers = TrunkLayerCount(g);
        var sliding = GgufMappingHelpers.SlidingLayers(g, layers, DefaultPattern, denseFirst: true);
        int denseLead = g.Int("leading_dense_block_count", 0);
        md[prefix + "attention.sliding_window"] = g.Int("attention.sliding_window");
        md[prefix + "attention.sliding_layers"] = sliding;
        md[prefix + "rope.layers"] = Enumerable.Range(0, layers).Select(il => sliding[il] != 0 || il < denseLead ? 1 : 0).ToArray();
        if (g.Int("expert_shared_count", 0) > 0 || g.Int("expert_shared_feed_forward_length", 0) > 0)
        {
            float routedScale = g.Float("expert_weights_scale", 0f) is var scale && scale != 0f ? scale : 1f;   // llama.cpp: 0 = unscaled
            md[prefix + "moe.weights_scale"] = AverageScale * routedScale;
            md[prefix + "moe.shared_scale"] = AverageScale;
        }
    }
}
