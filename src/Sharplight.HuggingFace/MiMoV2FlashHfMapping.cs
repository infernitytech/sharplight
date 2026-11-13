using System.Text.Json;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>mimo_v2_flash</c> (MiMo-V2-Flash): interleaved sliding-window layers (the first layer and every sixth are global)
/// with learned attention sinks and twice the key/value heads of the global layers, a value head size smaller than the key head size
/// (<c>v_head_dim</c>) and a constant factor on the attention values, partial rotary embeddings with a separate base for the sliding layers
/// (<c>rope_parameters</c> per layer type, or the flat <c>rope_theta</c> / <c>swa_rope_theta</c> / <c>partial_rotary_factor</c> keys of the published
/// config), and a leading dense layer followed by sigmoid-routed experts with selection bias and no shared expert (<c>mlp_layer_types</c>). Multi-token-prediction layers are not used.
/// </summary>
internal sealed class MiMoV2FlashHfMapping : HfMoeFamilyMapping
{
    private const string FullAttention = "full_attention";
    private const string SlidingAttention = "sliding_attention";
    private const int DefaultGlobalEvery = 6;
    private const double DefaultRopeBase = 10000.0;

    private static readonly IReadOnlyDictionary<string, string> s_layerTensors = With(LlamaLayerTensors,
        ("self_attn.sinks", "attention.sinks"),
        ("self_attn.attention_sink_bias", "attention.sinks"));

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["mimo_v2_flash"];

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> LayerTensors => s_layerTensors;

    /// <inheritdoc/>
    protected override int RopeDimension(HfConfig c, int headDim)
        => (int)(headDim * (HfMappingHelpers.OptionalNumber(LayerTypeRope(c, FullAttention), "partial_rotary_factor") ?? c.RopeDouble("partial_rotary_factor", "partial_rotary_factor") ?? 1.0));

    /// <inheritdoc/>
    protected override float RopeBase(HfConfig c)
        => (float)(HfMappingHelpers.OptionalNumber(LayerTypeRope(c, FullAttention), "rope_theta") ?? c.RopeDouble("rope_theta", "rope_theta") ?? DefaultRopeBase);

    /// <inheritdoc/>
    protected override MoeRouting Routing(HfConfig c)
        => new(c.Int("n_routed_experts", "num_local_experts"), c.Int("num_experts_per_tok"), c.Int("moe_intermediate_size"), "sigmoid",
            c.GetBool("norm_topk_prob") ?? true, (float)(c.GetDouble("routed_scaling_factor") ?? 1.0),
            c.GetInt("n_group") ?? 1, c.GetInt("topk_group") ?? 1);

    /// <inheritdoc/>
    protected override void AddFamilyHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        int layers = LayerCount(c);
        var sliding = SlidingFlags(c, layers);
        int kvHeads = c.Int("num_key_value_heads");
        md["decoder.attention.sliding_window"] = c.Int("sliding_window", "sliding_window_size");
        md["decoder.attention.sliding_layers"] = sliding;
        md["decoder.attention.layer_kv_head_counts"] = sliding.Select(flag => flag != 0 ? kvHeads * 2 : kvHeads).ToArray();
        md["decoder.attention.value_dim"] = c.Int("v_head_dim");
        md["decoder.rope.local_base"] = (float)(HfMappingHelpers.OptionalNumber(LayerTypeRope(c, SlidingAttention), "rope_theta") ?? c.GetDouble("swa_rope_theta") ?? DefaultRopeBase);
        if (c.GetDouble("attention_value_scale") is { } valueScale)
        {
            md["decoder.attention.value_scale"] = (float)valueScale;
        }
    }

    // layer_types (transformers), else hybrid_layer_pattern of the published config (1 = sliding window), else the first and every sixth layer global.
    private static int[] SlidingFlags(HfConfig c, int layers)
    {
        if (c.GetStringArray("layer_types") is { } layerTypes)
        {
            return HfMappingHelpers.SlidingFlags(layerTypes);
        }

        if (IntArray(c, "hybrid_layer_pattern") is { } pattern)
        {
            return pattern.Take(layers).ToArray();
        }

        return Enumerable.Range(0, layers).Select(il => il == 0 || (il + 1) % DefaultGlobalEvery == 0 ? 0 : 1).ToArray();
    }

    private static JsonElement? LayerTypeRope(HfConfig c, string layerType)
        => c.RopeParameters is { } parameters && parameters.TryGetProperty(layerType, out var element) && element.ValueKind == JsonValueKind.Object ? element : null;
}
