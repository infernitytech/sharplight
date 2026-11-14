using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>gemma4</c>, <c>gemma3n</c>, <c>muse-glimmer</c> and <c>flint</c>: the generic decoder plus per-layer head sizes, KV sharing, per-layer
/// embeddings, NoPE layers, AltUp streams, LAuReL, activation sparsity, ...
/// </summary>
internal sealed partial class DecoderGgufMapping
{
    private static bool IsExtended(string architecture) => architecture is "gemma4" or "gemma3n" or "muse-glimmer";

    /// <summary>Per-layer tensors of the extended architectures, GGUF suffix to canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_extended = new(StringComparer.Ordinal)
    {
        ["inp_gate.weight"] = "per_layer.gate.weight",
        ["proj.weight"] = "per_layer.projection.weight",
        ["post_norm.weight"] = "per_layer.norm.weight",
        ["layer_output_scale.weight"] = "output_scale",
        ["attn_gate.weight"] = "attention.gate.weight",
        // Gemma 3n: AltUp and LAuReL.
        ["altup_correct_scale.weight"] = "altup.correct_output_scale",
        ["altup_correct_coef.weight"] = "altup.correction_coefs.weight",
        ["altup_predict_coef.weight"] = "altup.prediction_coefs.weight",
        ["altup_router.weight"] = "altup.router.weight",
        ["altup_router_norm.weight"] = "altup.router_norm.weight",
        ["laurel_l.weight"] = "laurel.left.weight",
        ["laurel_r.weight"] = "laurel.right.weight",
        ["laurel_post_norm.weight"] = "laurel.norm.weight",
    };

    /// <summary>
    /// Gemma 3n stores the AltUp (un)embedding projections of streams 1 … S−1 stacked [S−1, H, H]; canonical weights are one [H, H] matrix per stream.
    /// </summary>
    private static IReadOnlyList<WeightInfo>? SplitAltUpProjections(string name, WeightInfo t)
    {
        string? canonical = name switch
        {
            "altup_proj.weight" => "altup.projections",
            "altup_unembd_proj.weight" => "altup.unembed_projections",
            _ => null,
        };
        if (canonical is null)
        {
            return null;
        }

        var matrix = new Shape(t.Shape[1], t.Shape[2]);
        return Enumerable.Range(0, (int)t.Shape[0].Fixed).Select(s => t.RowSlice("", s, 1).As($"{canonical}.{s}.weight", matrix)).ToList();
    }

    /// <summary>Per-layer-embedding tensors of the extended architectures that sit outside the layers.</summary>
    private static readonly Dictionary<string, string> s_extendedGlobal = new(StringComparer.Ordinal)
    {
        ["per_layer_token_embd.weight"] = "per_layer.embedding.weight",
        ["per_layer_model_proj.weight"] = "per_layer.model_projection.weight",
        ["per_layer_proj_norm.weight"] = "per_layer.projection_norm.weight",
    };

    /// <summary>A per-layer GGUF array (or a scalar repeated) as ints.</summary>
    private static int[] PerLayer(GgufMetadataView g, string key, int layerCount, int fallback)
    {
        if (g.Raw(key) is Array values && values.Length == layerCount)
        {
            return Enumerable.Range(0, layerCount).Select(i => Convert.ToInt32(values.GetValue(i))).ToArray();
        }

        return Enumerable.Repeat(g.Has(key) && g.Raw(key) is not Array ? g.Int(key) : fallback, layerCount).ToArray();
    }

    /// <summary>A per-layer GGUF boolean array as 0/1 flags; without a matching array every layer gets <paramref name="fallback"/>.</summary>
    private static int[] PerLayerFlags(GgufMetadataView g, string key, int layerCount, int fallback)
    {
        if (g.Raw(key) is Array values && values.Length == layerCount)
        {
            return Enumerable.Range(0, layerCount).Select(i => Convert.ToBoolean(values.GetValue(i)) ? 1 : 0).ToArray();
        }

        return Enumerable.Repeat(fallback, layerCount).ToArray();
    }

    /// <summary>
    /// Muse Glimmer: sandwich norms, per-head q/k RMS norms whose weights carry the q scale (the GGUF stores 3.87 for q), a sigmoid attention gate,
    /// sliding-window layers with rope alternating with full-attention NoPE layers (every <c>sliding_window_pattern</c>-th layer), weight-less norm of the
    /// embeddings, logits scaled before the soft cap. llama.cpp does not store the post-norm epsilon (1e-8 in the reference implementation).
    /// </summary>
    private IReadOnlyDictionary<string, object> MuseHyper(GgufMetadataView g)
    {
        string a = Architecture + ".";
        int layerCount = g.Int("block_count");
        int headCount = g.Int("attention.head_count");
        int embeddingLength = g.Int("embedding_length");
        int headDim = g.Int("attention.key_length", embeddingLength / headCount);
        var sliding = GgufMappingHelpers.SlidingLayers(layerCount, g.Int("attention.sliding_window_pattern", 4));
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = layerCount,
            [a + "hidden_size"] = embeddingLength,
            [a + "ffn_hidden_size"] = g.Int("feed_forward_length"),
            [a + "norm_eps"] = g.Float("attention.layer_norm_rms_epsilon"),
            [a + "post_norm_eps"] = 1e-8f,
            [a + "attention.head_count"] = headCount,
            [a + "attention.kv_head_count"] = g.Int("attention.head_count_kv", headCount),
            [a + "attention.head_dim"] = headDim,
            [a + "attention.sliding_window"] = g.Int("attention.sliding_window", 0),
            [a + "attention.sliding_layers"] = sliding,
            [a + "rope.dim"] = headDim,
            [a + "rope.base"] = g.Float("rope.freq_base", GgufMappingHelpers.DefaultRopeBase),
            [a + "rope.layers"] = sliding,
            [a + "rope.style"] = "neox",
            [a + "mlp.activation"] = "silu",
            [a + "embedding_norm"] = true,
        };
        GgufMappingHelpers.AddContextLength(md, a, g);

        if (g.Has("logit_scale"))
        {
            md[a + "logit_scale"] = g.Float("logit_scale");
        }

        if (g.Has("final_logit_softcapping"))
        {
            md[a + "final_logit_softcap"] = g.Float("final_logit_softcapping");
        }

        return md;
    }

    /// <summary>Flint: sliding-window layers with rope alternate with full-attention NoPE layers (rope_pattern), sigmoid-gated routed experts plus a shared expert, per-head attention sinks.</summary>
    private IReadOnlyDictionary<string, object> FlintHyper(GgufMetadataView g)
    {
        string a = Architecture + ".";
        int layerCount = g.Int("block_count");
        int headCount = g.Int("attention.head_count");
        int embeddingLength = g.Int("embedding_length");
        int headDim = g.Int("attention.key_length", embeddingLength / headCount);
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = layerCount,
            [a + "hidden_size"] = embeddingLength,
            [a + "ffn_hidden_size"] = g.Int("feed_forward_length"),
            [a + "norm_eps"] = g.Float("attention.layer_norm_rms_epsilon"),
            [a + "attention.head_count"] = headCount,
            [a + "attention.kv_head_count"] = g.Int("attention.head_count_kv", headCount),
            [a + "attention.head_dim"] = headDim,
            [a + "attention.sliding_window"] = g.Int("attention.sliding_window", 0),
            [a + "attention.sliding_layers"] = PerLayerFlags(g, "attention.sliding_window_pattern", layerCount, 0),
            [a + "rope.dim"] = headDim,
            [a + "rope.base"] = g.Float("rope.freq_base", GgufMappingHelpers.DefaultRopeBase),
            [a + "rope.layers"] = PerLayerFlags(g, "attention.rope_pattern", layerCount, 1),
            [a + "rope.style"] = "neox",
            [a + "mlp.activation"] = "silu",
            [a + "moe.expert_count"] = g.Int("expert_count"),
            [a + "moe.expert_used_count"] = g.Int("expert_used_count"),
            [a + "moe.ffn_hidden_size"] = g.Int("expert_feed_forward_length"),
            [a + "moe.normalize_weights"] = g.Bool("expert_weights_norm", false),
            [a + "moe.router_activation"] = g.Int("expert_gating_func", 1) == 2 ? "sigmoid" : "softmax",
        };
        if (g.Has("expert_weights_scale") && g.Float("expert_weights_scale") is var weightsScale && weightsScale != 1f)
        {
            md[a + "moe.weights_scale"] = weightsScale;
        }

        GgufMappingHelpers.AddContextLength(md, a, g);
        return md;
    }

    private IReadOnlyDictionary<string, object> Gemma4Hyper(GgufMetadataView g)
    {
        string a = Architecture + ".";
        int layerCount = g.Int("block_count");
        int headCount = g.Int("attention.head_count");
        int embeddingLength = g.Int("embedding_length");
        int globalDim = g.Int("attention.key_length");
        int localDim = g.Int("attention.key_length_swa", globalDim);

        // llama.cpp: sliding_window_pattern[il] = true for sliding layers.
        var pattern = g.Raw("attention.sliding_window_pattern") as Array ?? throw new InvalidDataException("gemma4: attention.sliding_window_pattern missing.");
        var sliding = Enumerable.Range(0, layerCount).Select(i => Convert.ToBoolean(pattern.GetValue(i)) ? 1 : 0).ToArray();
        var kvHeadCounts = PerLayer(g, "attention.head_count_kv", layerCount, headCount);
        var ffnLengths = PerLayer(g, "feed_forward_length", layerCount, 0);
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = layerCount,
            [a + "hidden_size"] = embeddingLength,
            [a + "ffn_hidden_size"] = ffnLengths.Min(),
            [a + "norm_eps"] = g.Float("attention.layer_norm_rms_epsilon"),
            [a + "attention.head_count"] = headCount,
            [a + "attention.kv_head_count"] = kvHeadCounts[0],
            [a + "attention.head_dim"] = localDim,
            [a + "attention.scale"] = 1f,
            [a + "attention.v_norm"] = true,
            [a + "attention.layer_head_dims"] = sliding.Select(s => s != 0 ? localDim : globalDim).ToArray(),
            [a + "attention.layer_kv_head_counts"] = kvHeadCounts,
            [a + "attention.sliding_window"] = g.Int("attention.sliding_window", 0),
            [a + "attention.sliding_layers"] = sliding,
            [a + "rope.dim"] = g.Int("rope.dimension_count", globalDim),
            [a + "rope.local_dim"] = g.Int("rope.dimension_count_swa", localDim),
            [a + "rope.base"] = g.Float("rope.freq_base", 1000000f),
            [a + "rope.local_base"] = g.Float("rope.freq_base_swa", GgufMappingHelpers.DefaultRopeBase),
            [a + "rope.style"] = "neox",
            [a + "mlp.activation"] = "gelu_tanh",
            [a + "embedding_scale"] = MathF.Sqrt(embeddingLength),
        };
        GgufMappingHelpers.AddContextLength(md, a, g);

        if (g.Has("final_logit_softcapping"))
        {
            md[a + "final_logit_softcap"] = g.Float("final_logit_softcapping");
        }

        int sharedKvLayers = g.Int("attention.shared_kv_layers", 0);
        if (sharedKvLayers > 0)
        {
            md[a + "attention.kv_source_layers"] = KvSourceLayers(layerCount, sharedKvLayers, sliding);
        }

        if (g.Int("embedding_length_per_layer_input", 0) is var perLayerSize and > 0)
        {
            md[a + "per_layer.size"] = perLayerSize;
        }

        if (g.Int("altup.num_inputs", 0) is var streams and > 1)
        {
            if (g.Int("altup.active_idx", 0) != 0)
            {
                throw new NotSupportedException("gemma3n: only AltUp with active stream 0 is supported.");
            }

            md[a + "altup.streams"] = streams;
        }

        // llama.cpp stores the gate cutoff in standard deviations, Φ⁻¹(sparsity) (−∞ for dense layers); canonical is the sparsity itself.
        if (g.Raw("activation_sparsity_scale") is Array cutoffs && cutoffs.Length == layerCount)
        {
            var sparsity = Enumerable.Range(0, layerCount)
                .Select(i => Convert.ToDouble(cutoffs.GetValue(i)) is var z && double.IsFinite(z) ? (float)NormalDistribution.Cdf(z) : 0f).ToArray();
            if (sparsity.Any(s => s > 0))
            {
                md[a + "mlp.activation_sparsity"] = sparsity;
            }
        }

        return md;
    }

    /// <summary>
    /// For KV-sharing models the last <paramref name="sharedKvLayers"/> layers reuse the KV cache of the last earlier
    /// layer with the same attention kind (sliding or global); every other layer is its own source.
    /// </summary>
    private static int[] KvSourceLayers(int layerCount, int sharedKvLayers, int[] sliding)
    {
        int firstSharedLayer = layerCount - sharedKvLayers;
        var source = new int[layerCount];
        for (int il = 0; il < layerCount; il++)
        {
            source[il] = il < firstSharedLayer ? il : Enumerable.Range(0, firstSharedLayer).Last(j => sliding[j] == sliding[il]);
        }

        return source;
    }
}
