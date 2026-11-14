using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// Gemma 4, Gemma 3n, Muse Glimmer and Flint: the generic decoder plus per-layer head sizes, KV sharing, per-layer embeddings, NoPE layers,
/// AltUp streams, LAuReL and activation sparsity, ...
/// </summary>
internal sealed partial class DecoderHfMapping
{
    private static bool IsGemma4(string? modelType) => modelType is "gemma4" or "gemma4_text" or "gemma4_unified" or "gemma4_unified_text";

    private static bool IsGemma3n(string? modelType) => modelType is "gemma3n" or "gemma3n_text";

    private static bool IsGemma4OrMuse(string? modelType) => IsGemma4(modelType) || IsGemma3n(modelType) || modelType == "muse_glimmer";

    /// <summary>Vision / audio towers and their projectors live in other canonical sources.</summary>
    private static bool IsNonTextTensor(string name)
        => name.StartsWith("model.vision_tower.", StringComparison.Ordinal)
        || name.StartsWith("model.audio_tower.", StringComparison.Ordinal)
        || name.StartsWith("model.embed_vision.", StringComparison.Ordinal)
        || name.StartsWith("model.embed_audio.", StringComparison.Ordinal)
        || name.StartsWith("model.vision_", StringComparison.Ordinal)
        || name.StartsWith("model.visual.", StringComparison.Ordinal);

    /// <summary>Per-layer tensors of the extended architectures, HF suffix to canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_extended = new(StringComparer.Ordinal)
    {
        ["self_attn.gate_proj.weight"] = "attention.gate.weight",
        ["per_layer_input_gate.weight"] = "per_layer.gate.weight",
        ["per_layer_projection.weight"] = "per_layer.projection.weight",
        ["post_per_layer_input_norm.weight"] = "per_layer.norm.weight",
        ["layer_scalar"] = "output_scale",
        ["altup.router_norm.weight"] = "altup.router_norm.weight",
        ["altup.modality_router.weight"] = "altup.router.weight",
        ["altup.prediction_coefs.weight"] = "altup.prediction_coefs.weight",
        ["altup.correction_coefs.weight"] = "altup.correction_coefs.weight",
        ["laurel.linear_left.weight"] = "laurel.left.weight",
        ["laurel.linear_right.weight"] = "laurel.right.weight",
        ["laurel.post_laurel_norm.weight"] = "laurel.norm.weight",
    };

    /// <summary>The <c>rope_parameters</c> object of one layer type (<c>full_attention</c> / <c>sliding_attention</c>), or null.</summary>
    private static JsonElement? LayerTypeRope(HfConfig c, string layerType)
        => c.RopeParameters is { } ropeParameters && ropeParameters.TryGetProperty(layerType, out var element) && element.ValueKind == JsonValueKind.Object ? element : null;

    private static string[] LayerTypes(HfConfig c, int layerCount)
        => c.GetStringArray("layer_types") ?? Enumerable.Repeat("full_attention", layerCount).ToArray();

    /// <summary>Per-frequency divisors of the global layers' rope (Gemma 4 proportional rotary), or null.</summary>
    private static float[]? ExtendedRopeFactors(HfConfig c)
    {
        if (!IsGemma4(c.ModelType))
        {
            return null;
        }

        if (LayerTypeRope(c, "full_attention") is not { } full || !full.TryGetProperty("rope_type", out var ropeType) || ropeType.GetString() != "proportional")
        {
            return null;
        }

        int dim = c.GetInt("global_head_dim") ?? c.Int("head_dim");
        int angles = (int)(HfMappingHelpers.NumberOr(full, "partial_rotary_factor", 1.0) * dim / 2);
        return RopeScaling.Proportional(dim, angles, (float)HfMappingHelpers.NumberOr(full, "factor", 1.0)).Factors;
    }

    /// <summary>Muse Glimmer's q/k norms have no weights; canonical weights make them explicit (k: 1, q: qk_scale_factor, the scale applied to q after its norm).</summary>
    private static IEnumerable<WeightInfo> MuseQkNorms(HfConfig c)
    {
        int layerCount = c.Int("num_hidden_layers");
        int headDim = HeadDim(c);
        float qScale = (float)(c.GetDouble("qk_scale_factor") ?? 1.0);
        for (int il = 0; il < layerCount; il++)
        {
            var q = new float[headDim];
            Array.Fill(q, qScale);
            var k = new float[headDim];
            Array.Fill(k, 1f);
            yield return HfMappingHelpers.F32Weight($"layers.{il}.attention.q_norm.weight", q);
            yield return HfMappingHelpers.F32Weight($"layers.{il}.attention.k_norm.weight", k);
        }
    }

    /// <summary>
    /// Flint: Qwen3-MoE-style routed experts with sigmoid router scores, an always-on shared expert, per-head attention sinks, sliding-window layers
    /// that carry rope alternating with full-attention NoPE layers, two leading dense layers.
    /// </summary>
    private IReadOnlyDictionary<string, object> FlintHyper(HfConfig c)
    {
        string a = Architecture + ".";
        int layerCount = c.Int("num_hidden_layers");
        int headCount = c.Int("num_attention_heads");
        int hidden = c.Int("hidden_size");
        int headDim = c.GetInt("head_dim") ?? hidden / headCount;
        var sliding = HfMappingHelpers.SlidingFlags(LayerTypes(c, layerCount));
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = layerCount,
            [a + "hidden_size"] = hidden,
            [a + "ffn_hidden_size"] = c.Int("intermediate_size"),
            [a + "norm_eps"] = (float)(c.GetDouble("rms_norm_eps") ?? 1e-6),
            [a + "attention.head_count"] = headCount,
            [a + "attention.kv_head_count"] = c.GetInt("num_key_value_heads") ?? headCount,
            [a + "attention.head_dim"] = headDim,
            [a + "attention.sliding_window"] = c.GetInt("sliding_window") ?? 0,
            [a + "attention.sliding_layers"] = sliding,
            [a + "rope.dim"] = headDim,
            [a + "rope.base"] = RopeBase(c),
            [a + "rope.layers"] = sliding,   // only the sliding-window layers are rotary; full-attention layers use no positional encoding
            [a + "rope.style"] = "neox",
            [a + "mlp.activation"] = Activation(c),
            [a + "moe.expert_count"] = c.Int("num_experts"),
            [a + "moe.expert_used_count"] = c.Int("num_experts_per_tok"),
            [a + "moe.ffn_hidden_size"] = c.Int("moe_intermediate_size"),
            [a + "moe.normalize_weights"] = c.GetBool("norm_topk_prob") ?? false,
            [a + "moe.router_activation"] = c.GetString("router_score_function") ?? "sigmoid",
        };
        HfMappingHelpers.AddContextLength(md, a, c);
        return md;
    }

    private IReadOnlyDictionary<string, object> Gemma4Hyper(HfConfig c)
    {
        if (c.GetBool("enable_moe_block") == true)
        {
            throw new NotSupportedException("Gemma 4 mixture-of-experts blocks are not supported yet.");
        }

        string a = Architecture + ".";
        int layerCount = c.Int("num_hidden_layers");
        int headCount = c.Int("num_attention_heads");
        int hidden = c.Int("hidden_size");
        int headDim = c.Int("head_dim");
        int globalDim = c.GetInt("global_head_dim") ?? headDim;
        int kvHeadCount = c.GetInt("num_key_value_heads") ?? headCount;
        int globalKvHeadCount = c.GetBool("attention_k_eq_v") == true ? c.GetInt("num_global_key_value_heads") ?? kvHeadCount : kvHeadCount;
        var types = LayerTypes(c, layerCount);
        var fullRope = LayerTypeRope(c, "full_attention");
        var slidingRope = LayerTypeRope(c, "sliding_attention");
        // Gemma 3n lists the MLP width per layer; hub configs carry rope_theta / rope_local_base_freq instead of per-type rope parameters.
        int ffnLength = c.Get("intermediate_size") is { ValueKind: JsonValueKind.Array } widths ? widths.EnumerateArray().Max(e => e.GetInt32()) : c.Int("intermediate_size");
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = layerCount,
            [a + "hidden_size"] = hidden,
            [a + "ffn_hidden_size"] = ffnLength,
            [a + "norm_eps"] = (float)(c.GetDouble("rms_norm_eps") ?? 1e-6),
            [a + "attention.head_count"] = headCount,
            [a + "attention.kv_head_count"] = kvHeadCount,
            [a + "attention.head_dim"] = headDim,
            [a + "attention.scale"] = 1f,
            [a + "attention.v_norm"] = true,
            [a + "attention.layer_head_dims"] = types.Select(t => t == "full_attention" ? globalDim : headDim).ToArray(),
            [a + "attention.layer_kv_head_counts"] = types.Select(t => t == "full_attention" ? globalKvHeadCount : kvHeadCount).ToArray(),
            [a + "attention.sliding_window"] = c.GetInt("sliding_window") ?? 0,
            [a + "attention.sliding_layers"] = HfMappingHelpers.SlidingFlags(types),
            [a + "rope.dim"] = globalDim,
            [a + "rope.local_dim"] = headDim,
            [a + "rope.base"] = (float)HfMappingHelpers.NumberOr(fullRope, "rope_theta", c.GetDouble("rope_theta") ?? 1000000.0),
            [a + "rope.local_base"] = (float)HfMappingHelpers.NumberOr(slidingRope, "rope_theta", c.GetDouble("rope_local_base_freq") ?? DefaultRopeBase),
            [a + "rope.style"] = "neox",
            [a + "mlp.activation"] = Activation(c),
            [a + "embedding_scale"] = MathF.Sqrt(hidden),
        };
        HfMappingHelpers.AddContextLength(md, a, c);

        if (c.GetDouble("final_logit_softcapping") is { } softcap && softcap > 0)
        {
            md[a + "final_logit_softcap"] = (float)softcap;
        }

        int sharedKvLayers = c.GetInt("num_kv_shared_layers") ?? 0;
        if (sharedKvLayers > 0)
        {
            md[a + "attention.kv_source_layers"] = KvSourceLayers(layerCount, sharedKvLayers, types);
        }

        if (c.GetInt("hidden_size_per_layer_input") is { } perLayerSize and > 0)
        {
            md[a + "per_layer.size"] = perLayerSize;
        }

        if (c.GetInt("altup_num_inputs") is { } streams and > 1)
        {
            if ((c.GetInt("altup_active_idx") ?? 0) != 0)
            {
                throw new NotSupportedException("Gemma 3n: only AltUp with active stream 0 is supported.");
            }

            md[a + "altup.streams"] = streams;
        }

        if (c.Get("activation_sparsity_pattern") is { ValueKind: JsonValueKind.Array } sparsity && sparsity.EnumerateArray().Any(e => e.GetDouble() > 0))
        {
            md[a + "mlp.activation_sparsity"] = sparsity.EnumerateArray().Select(e => (float)e.GetDouble()).ToArray();
        }

        return md;
    }

    /// <summary>
    /// For KV-sharing models the last <paramref name="sharedKvLayers"/> layers reuse the KV cache of the last earlier
    /// layer of the same type; every other layer is its own source.
    /// </summary>
    private static int[] KvSourceLayers(int layerCount, int sharedKvLayers, string[] layerTypes)
    {
        int firstSharedLayer = layerCount - sharedKvLayers;
        var source = new int[layerCount];
        for (int il = 0; il < layerCount; il++)
        {
            source[il] = il < firstSharedLayer ? il : Enumerable.Range(0, firstSharedLayer).Last(j => layerTypes[j] == layerTypes[il]);
        }

        return source;
    }

    private IReadOnlyDictionary<string, object> MuseHyper(HfConfig c)
    {
        string a = Architecture + ".";
        int layerCount = c.Int("num_hidden_layers");
        int headCount = c.Int("num_attention_heads");
        int hidden = c.Int("hidden_size");
        int headDim = c.GetInt("head_dim") ?? hidden / headCount;
        var types = LayerTypes(c, layerCount);
        double ropeTheta = HfMappingHelpers.NumberOr(c.RopeParameters, "rope_theta", DefaultRopeBase);
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = layerCount,
            [a + "hidden_size"] = hidden,
            [a + "ffn_hidden_size"] = c.Int("intermediate_size"),
            [a + "norm_eps"] = (float)(c.GetDouble("rms_norm_eps") ?? 1e-5),
            [a + "post_norm_eps"] = (float)(c.GetDouble("post_norm_eps") ?? 1e-5),
            [a + "attention.head_count"] = headCount,
            [a + "attention.kv_head_count"] = c.GetInt("num_key_value_heads") ?? headCount,
            [a + "attention.head_dim"] = headDim,
            [a + "attention.sliding_window"] = c.GetInt("sliding_window") ?? 0,
            [a + "attention.sliding_layers"] = HfMappingHelpers.SlidingFlags(types),
            [a + "rope.dim"] = headDim,
            [a + "rope.base"] = (float)ropeTheta,
            [a + "rope.style"] = "neox",
            [a + "mlp.activation"] = Activation(c),
            [a + "embedding_norm"] = true,
        };
        if (c.Get("layer_rope_theta") is { ValueKind: JsonValueKind.Array } layerRopeTheta)
        {
            md[a + "rope.layers"] = layerRopeTheta.EnumerateArray().Select(e => e.GetDouble() != 0 ? 1 : 0).ToArray();
        }

        HfMappingHelpers.AddContextLength(md, a, c);

        if (c.GetDouble("output_multiplier") is { } outputMultiplier && outputMultiplier != 1.0)
        {
            md[a + "logit_scale"] = (float)outputMultiplier;
        }

        if (c.GetDouble("final_logit_softcapping") is { } softcap && softcap > 0)
        {
            md[a + "final_logit_softcap"] = (float)softcap;
        }

        return md;
    }
}
