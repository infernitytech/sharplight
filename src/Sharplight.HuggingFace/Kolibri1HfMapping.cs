namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>kolibri1</c> (Kolibri-1): sandwich norms around both sub-layers (<c>post_self_attn_layernorm</c>,
/// <c>post_mlp_layernorm</c>), per-head q/k norms, sliding-window layers from <c>layer_types</c> that alone carry the rotary embedding (the
/// global layers are NoPE when the model has a window), and sigmoid-routed experts selected by the top-k of the logits plus
/// <c>e_score_correction_bias</c> (weights from the unbiased logits, renormalized with <c>norm_topk_prob</c>) plus a shared expert.
/// </summary>
internal sealed class Kolibri1HfMapping : HfMoeFamilyMapping
{
    private static readonly IReadOnlyDictionary<string, string> s_layerTensors = With(LlamaLayerTensors,
        ("post_self_attn_layernorm.weight", "attention.out_norm.weight"),
        ("post_mlp_layernorm.weight", "mlp.out_norm.weight"),
        // The original checkpoint names (transformers saves them back this way).
        ("post_attn_norm.weight", "attention.out_norm.weight"),
        ("post_ffn_norm.weight", "mlp.out_norm.weight"),
        ("moe.router.expert_bias", "mlp.router_bias"));

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["kolibri1"];

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> LayerTensors => s_layerTensors;

    /// <inheritdoc/>
    /// <remarks>Every layer is a MoE layer; the widest MLP is the shared expert.</remarks>
    protected override int FeedForwardSize(HfConfig c) => c.Int("shared_expert_intermediate_size");

    /// <inheritdoc/>
    protected override MoeRouting Routing(HfConfig c)
        => new(c.Int("num_local_experts", "num_experts"), c.Int("num_experts_per_tok"), c.Int("moe_intermediate_size"), "sigmoid",
            Normalize: c.GetBool("norm_topk_prob") ?? false);

    /// <inheritdoc/>
    protected override void AddFamilyHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        md["decoder.moe.selection_bias_on_logits"] = true;
        if (c.GetInt("sliding_window") is > 0 and var window)
        {
            var sliding = c.GetStringArray("layer_types") is { } layerTypes
                ? HfMappingHelpers.SlidingFlags(layerTypes)
                : Enumerable.Range(0, LayerCount(c)).Select(il => (il + 1) % 5 != 0 ? 1 : 0).ToArray();
            md["decoder.attention.sliding_window"] = window;
            md["decoder.attention.sliding_layers"] = sliding;
            md["decoder.rope.layers"] = sliding;
        }
    }
}
