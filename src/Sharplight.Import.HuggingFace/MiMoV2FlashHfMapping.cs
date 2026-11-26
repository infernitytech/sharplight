using System.Globalization;
using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Hugging Face <c>mimo_v2_flash</c> (MiMo-V2-Flash): interleaved sliding-window layers (the first layer and every sixth are global)
/// with learned attention sinks and twice the key/value heads of the global layers, a value head size smaller than the key head size
/// (<c>v_head_dim</c>) and a constant factor on the attention values, partial rotary embeddings with a separate base for the sliding layers
/// (<c>rope_parameters</c> per layer type, or the flat <c>rope_theta</c> / <c>swa_rope_theta</c> / <c>partial_rotary_factor</c> keys of the published
/// config), and a leading dense layer followed by sigmoid-routed experts with selection bias and no shared expert (<c>mlp_layer_types</c>). Multi-token-prediction layers are not used.
/// Also <c>mimo_v2</c> (the MiMo-V2.x remote code, e.g. MiMo-V2.6-Pro): the same layers, with the query / key / value projections fused into
/// <c>qkv_proj</c> when <c>attention_projection_layout</c> is <c>fused_qkv</c>, the sliding layers' key/value head count in
/// <c>swa_num_key_value_heads</c>, and a vision tower, audio encoder and speech embeddings that the text import skips.
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
    public override IReadOnlyList<string> ModelTypes { get; } = ["mimo_v2_flash", "mimo_v2"];

    // Tensors of the multimodal mimo_v2 checkpoints outside the text decoder, and the multi-token-prediction layers.
    private static readonly string[] s_nonTextPrefixes = ["model.mtp.", "visual.", "audio_encoder.", "speech_embeddings."];

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> LayerTensors => s_layerTensors;

    /// <inheritdoc/>
    public override IEnumerable<string> Validate(HfConfig config)
    {
        foreach (var (swaKey, key) in new[] { ("swa_head_dim", "head_dim"), ("swa_v_head_dim", "v_head_dim"), ("swa_num_attention_heads", "num_attention_heads") })
        {
            if (config.GetInt(swaKey) is { } swa && config.GetInt(key) is { } full && swa != full)
            {
                yield return $"{config.ModelType}: sliding layers with {swaKey} {swa} other than {key} {full} are not supported";
            }
        }
    }

    /// <inheritdoc/>
    protected override bool IsIgnored(string name)
        => base.IsIgnored(name) || s_nonTextPrefixes.Any(prefix => name.StartsWith(prefix, StringComparison.Ordinal));

    /// <inheritdoc/>
    protected override bool TryMapFamilyTensor(HfConfig c, string layerPrefix, string suffix, WeightInfo w, HfWeightCollector collector)
    {
        if (suffix != "self_attn.qkv_proj.weight")
        {
            return false;
        }

        int layer = int.Parse(layerPrefix["layers.".Length..^1], CultureInfo.InvariantCulture);
        int headDim = HeadDimOf(c);
        int queryRows = HeadCount(c) * headDim;
        int kvHeads = LayerKvHeadCounts(c)[layer];
        int keyRows = kvHeads * headDim;
        int valueRows = kvHeads * c.Int("v_head_dim");
        int chunks = PreShardedChunks(c, kvHeads);
        long chunkRows = (queryRows + keyRows + valueRows) / chunks;
        if (w.Shape[0].Fixed != chunks * chunkRows)
        {
            throw new InvalidDataException($"{layerPrefix}self_attn.qkv_proj.weight: {w.Shape[0].Fixed} rows are not {chunks} slices of query, key and value rows.");
        }

        (long, long)[] Part(long offset, long rows) => [.. Enumerable.Range(0, chunks).Select(i => (i * chunkRows + offset / chunks, rows / chunks))];
        collector.Add(w.GatherRows(layerPrefix + "attention.q.weight", Part(0, queryRows)));
        collector.Add(w.GatherRows(layerPrefix + "attention.k.weight", Part(queryRows, keyRows)));
        collector.Add(w.GatherRows(layerPrefix + "attention.v.weight", Part(queryRows + keyRows, valueRows)));
        return true;
    }

    /// <summary>
    /// The slices the fused <c>qkv_proj</c> is stored in: 1 ([Q | K | V]), or for block-quantized FP8 checkpoints (MiMo-V2.6-Pro) one slice per
    /// key-value head of <c>num_key_value_heads</c>, [Q_0 | K_0 | V_0 | Q_1 | K_1 | V_1 | ...], each quantized on its own (pre-sharded for
    /// tensor parallelism, as vLLM's <c>mimo_v2</c> loader reads them).
    /// </summary>
    private int PreShardedChunks(HfConfig c, int layerKvHeads)
    {
        if (!c.TryGetQuantization("quantization_config", out var quantization) || !quantization.TryGetProperty("weight_block_size", out var block)
            || block.ValueKind != JsonValueKind.Array)
        {
            return 1;
        }

        int chunks = c.Int("num_key_value_heads");
        return HeadCount(c) % chunks == 0 && layerKvHeads % chunks == 0 ? chunks : 1;
    }

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
        md["decoder.attention.sliding_window"] = c.Int("sliding_window", "sliding_window_size");
        md["decoder.attention.sliding_layers"] = SlidingFlags(c, layers);
        md["decoder.attention.layer_kv_head_counts"] = LayerKvHeadCounts(c);
        md["decoder.attention.value_dim"] = c.Int("v_head_dim");
        md["decoder.rope.local_base"] = (float)(HfMappingHelpers.OptionalNumber(LayerTypeRope(c, SlidingAttention), "rope_theta") ?? c.GetDouble("swa_rope_theta") ?? DefaultRopeBase);
        if (c.GetDouble("attention_value_scale") is { } valueScale)
        {
            md["decoder.attention.value_scale"] = (float)valueScale;
        }
    }

    // Key/value heads per layer: swa_num_key_value_heads on the sliding layers (twice the global layers' when absent, as in MiMo-V2-Flash).
    private int[] LayerKvHeadCounts(HfConfig c)
    {
        int kvHeads = c.Int("num_key_value_heads");
        int slidingKvHeads = c.GetInt("swa_num_key_value_heads") ?? kvHeads * 2;
        return SlidingFlags(c, LayerCount(c)).Select(flag => flag != 0 ? slidingKvHeads : kvHeads).ToArray();
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
