namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>afmoe</c> (Arcee Trinity): sandwich norms around both sub-layers, per-head q/k norms and a sigmoid attention output
/// gate; sliding-window layers (every <c>global_attn_every_n_layers</c>-th layer is global) are the only ones with a rotary embedding;
/// leading dense layers, then a sigmoid router with selection bias (<c>expert_bias</c>), always renormalized and scaled by
/// <c>route_scale</c>, plus shared experts. With <c>mup_enabled</c> the token embeddings are scaled by the square root of the hidden size.
/// </summary>
internal sealed class AfmoeHfMapping : HfMoeFamilyMapping
{
    private const int DefaultGlobalEvery = 4;

    private static readonly IReadOnlyDictionary<string, string> s_layerTensors = With(LlamaLayerTensors,
        ("self_attn.gate_proj.weight", "attention.gate.weight"),
        ("post_attention_layernorm.weight", "attention.out_norm.weight"),
        ("pre_mlp_layernorm.weight", "post_attention_norm.weight"),
        ("post_mlp_layernorm.weight", "mlp.out_norm.weight"));

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["afmoe"];

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> LayerTensors => s_layerTensors;

    /// <inheritdoc/>
    protected override MoeRouting Routing(HfConfig c)
        => new(c.Int("num_experts"), c.Int("num_experts_per_tok"), c.Int("moe_intermediate_size"), "sigmoid",
            Normalize: true, Scale: (float)(c.GetDouble("route_scale") ?? 1.0));

    /// <inheritdoc/>
    protected override void AddFamilyHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        int every = c.GetInt("global_attn_every_n_layers") ?? DefaultGlobalEvery;
        var sliding = c.GetStringArray("layer_types") is { } layerTypes
            ? HfMappingHelpers.SlidingFlags(layerTypes)
            : Enumerable.Range(0, LayerCount(c)).Select(il => (il + 1) % every != 0 ? 1 : 0).ToArray();
        if (c.GetInt("sliding_window") is > 0 and var window && sliding.Any(flag => flag != 0))
        {
            md["decoder.attention.sliding_window"] = window;
            md["decoder.attention.sliding_layers"] = sliding;
        }

        md["decoder.rope.layers"] = sliding;
        if (c.GetBool("mup_enabled") ?? false)
        {
            md["decoder.embedding_scale"] = MathF.Sqrt(HiddenSize(c));
        }
    }
}
