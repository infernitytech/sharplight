using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>deepseek4</c> (DeepSeek-V4) → canonical <c>latent_decoder</c> with compressed-KV attention (<c>attention.compression.*</c>,
/// vocabulary on <c>LatentDecoderAdapter</c>): per-layer <c>attention.compress_ratios</c> (0 = sliding window only, 4 = overlapping windows with
/// the lightning indexer, 128 = dense compressed entries), <c>attention.sliding_window</c>, the grouped low-rank output
/// (<c>attention.output_group_count</c>), the compressed layers' rope base <c>attention.compress_rope_freq_base</c> with the file's yarn scaling
/// (the plain base and no scaling on the sliding-only layers, as llama.cpp), hyper-connections with Sinkhorn mixing, sqrt-softplus routing with the
/// hash-routed first <c>hash_layer_count</c> layers (<c>ffn_gate_tid2eid</c>) and clamped SwiGLU experts and shared expert. The NextN layers are
/// not imported.
/// </summary>
internal sealed class DeepSeek4GgufMapping : IGgufArchitectureMapping
{
    // llama.cpp expert_gating_func: sqrt(softplus(logit)).
    private const int GatingSqrtSoftplus = 4;

    /// <inheritdoc/>
    public IReadOnlyList<string> GgufArchitectures { get; } = ["deepseek4"];

    /// <inheritdoc/>
    public string Architecture => "latent_decoder";

    private static int TrunkLayers(GgufMetadataView g) => g.Int("block_count") - g.Int("nextn_predict_layers", 0);

    /// <inheritdoc/>
    /// <exception cref="InvalidDataException">The routing is not sqrt-softplus.</exception>
    public IReadOnlyDictionary<string, object> MapHyperParameters(GgufMetadataView g)
    {
        string a = Architecture + ".";
        int layers = TrunkLayers(g);
        if (g.Int("expert_gating_func", GatingSqrtSoftplus) != GatingSqrtSoftplus)
        {
            throw new InvalidDataException("deepseek4: only sqrt-softplus expert gating is supported.");
        }

        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = layers,
            [a + "hidden_size"] = g.Int("embedding_length"),
            [a + "norm_eps"] = g.Float("attention.layer_norm_rms_epsilon"),
            [a + "attention.head_count"] = g.Int("attention.head_count"),
            [a + "attention.q_rank"] = g.Int("attention.q_lora_rank"),
            [a + "attention.rope_dim"] = g.Int("rope.dimension_count"),
            [a + "attention.compression.ratios"] = GgufMappingHelpers.PerLayer(g, "attention.compress_ratios", layers),
            [a + "attention.compression.window"] = g.Int("attention.sliding_window"),
            [a + "attention.compression.top_k"] = g.Int("attention.indexer.top_k", 0),
            [a + "attention.compression.output_groups"] = g.Int("attention.output_group_count"),
            [a + "rope.base"] = g.Float("rope.freq_base", GgufMappingHelpers.DefaultRopeBase),
            [a + "rope.base_compressed"] = g.Float("attention.compress_rope_freq_base"),
            [a + "rope.style"] = "gptj",
            [a + "moe.expert_count"] = g.Int("expert_count"),
            [a + "moe.expert_used_count"] = g.Int("expert_used_count"),
            [a + "moe.router_activation"] = "sqrt_softplus",
            [a + "moe.normalize_weights"] = g.Bool("expert_weights_norm", false),
            [a + "moe.weights_scale"] = g.Float("expert_weights_scale", 1f),
            [a + "hyper_connections.streams"] = g.Int("hyper_connection.count"),
            [a + "hyper_connections.eps"] = g.Float("hyper_connection.epsilon", 1e-6f),
            [a + "hyper_connections.magnitude"] = 2f,
            [a + "hyper_connections.post_offset"] = 0f,
            [a + "hyper_connections.sinkhorn_iterations"] = g.Int("hyper_connection.sinkhorn_iterations", 20),
        };
        GgufMappingHelpers.AddContextLength(md, a, g);
        if (g.Has("swiglu_clamp_exp"))
        {
            md[a + "moe.swiglu_clamp"] = GgufMappingHelpers.PerLayerFloat(g, "swiglu_clamp_exp", layers);
            md[a + "moe.swiglu_clamp_input"] = true;
            md[a + "mlp.swiglu_clamp"] = GgufMappingHelpers.PerLayerFloat(g, g.Has("swiglu_clamp_shexp") ? "swiglu_clamp_shexp" : "swiglu_clamp_exp", layers);
            md[a + "mlp.swiglu_clamp_input"] = true;
        }

        return md;
    }

    /// <summary>Global tensors besides the embedding, output norm and head.</summary>
    private static readonly Dictionary<string, string> s_global = new(StringComparer.Ordinal)
    {
        ["output_hc_fn.weight"] = "output_hc.fn.weight",
        ["output_hc_base.weight"] = "output_hc.base",
        ["output_hc_scale.weight"] = "output_hc.scale",
    };

    /// <summary>Per-layer tensors, GGUF suffix to canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_layer = new(StringComparer.Ordinal)
    {
        ["attn_norm.weight"] = "input_norm.weight",
        ["ffn_norm.weight"] = "post_attention_norm.weight",
        ["attn_sinks.weight"] = "attention.sinks",
        ["attn_q_a.weight"] = "attention.q_down.weight",
        ["attn_q_a_norm.weight"] = "attention.q_down_norm.weight",
        ["attn_q_b.weight"] = "attention.q_up.weight",
        ["attn_kv.weight"] = "attention.kv.weight",
        ["attn_kv_a_norm.weight"] = "attention.kv_norm.weight",
        ["attn_output_a.weight"] = "attention.o_group.weight",
        ["attn_output_b.weight"] = "attention.o.weight",
        ["attn_compressor_kv.weight"] = "attention.compressor.kv.weight",
        ["attn_compressor_gate.weight"] = "attention.compressor.gate.weight",
        ["attn_compressor_ape.weight"] = "attention.compressor.position_bias",
        ["attn_compressor_norm.weight"] = "attention.compressor.norm.weight",
        ["indexer.attn_q_b.weight"] = "attention.indexer.q_up.weight",
        ["indexer.proj.weight"] = "attention.indexer.head_weights.weight",
        ["indexer_compressor_kv.weight"] = "attention.indexer.compressor.kv.weight",
        ["indexer_compressor_gate.weight"] = "attention.indexer.compressor.gate.weight",
        ["indexer_compressor_ape.weight"] = "attention.indexer.compressor.position_bias",
        ["indexer_compressor_norm.weight"] = "attention.indexer.compressor.norm.weight",
        ["hc_attn_fn.weight"] = "attention_hc.fn.weight",
        ["hc_attn_base.weight"] = "attention_hc.base",
        ["hc_attn_scale.weight"] = "attention_hc.scale",
        ["hc_ffn_fn.weight"] = "mlp_hc.fn.weight",
        ["hc_ffn_base.weight"] = "mlp_hc.base",
        ["hc_ffn_scale.weight"] = "mlp_hc.scale",
        ["ffn_gate_inp.weight"] = "mlp.router.weight",
        ["exp_probs_b.bias"] = "mlp.router_bias",
        ["ffn_gate_tid2eid.weight"] = "mlp.router.token_experts",
        ["ffn_gate_exps.weight"] = "mlp.experts.gate.weight",
        ["ffn_up_exps.weight"] = "mlp.experts.up.weight",
        ["ffn_down_exps.weight"] = "mlp.experts.down.weight",
        ["ffn_gate_shexp.weight"] = "mlp.shared.gate.weight",
        ["ffn_up_shexp.weight"] = "mlp.shared.up.weight",
        ["ffn_down_shexp.weight"] = "mlp.shared.down.weight",
    };

    /// <inheritdoc/>
    public string? MapWeightName(string name, GgufMetadataView g)
    {
        if (GgufTensorNames.MapGlobal(name) is { } globalName)
        {
            return globalName;
        }

        if (s_global.TryGetValue(name, out var hc))
        {
            return hc;
        }

        var m = GgufTensorNames.Block().Match(name);
        if (!m.Success || int.Parse(m.Groups[1].Value) >= TrunkLayers(g))
        {
            return null;
        }

        return s_layer.TryGetValue(m.Groups[2].Value, out var canonical) ? $"layers.{m.Groups[1].Value}.{canonical}" : null;
    }

    /// <inheritdoc/>
    public WeightInfo MapWeight(WeightInfo weight, GgufMetadataView g)
    {
        if (!weight.Name.EndsWith(".attention.o_group.weight", StringComparison.Ordinal))
        {
            return weight;
        }

        // [G·R, H·D/G] → [G, R, H·D/G]: one "expert" per output group.
        int groups = g.Int("attention.output_group_count");
        return weight.As(weight.Name, new Shape(groups, weight.Shape[0].Fixed / groups, weight.Shape[1].Fixed));
    }

    /// <inheritdoc/>
    public IEnumerable<WeightInfo> SyntheticWeights(GgufMetadataView g, IReadOnlyDictionary<string, WeightInfo> weights)
    {
        if (g.Raw("rope.scaling.type") is not "yarn")
        {
            yield break;
        }

        // The compressed layers' yarn; llama.cpp cancels yarn's cos/sin magnitude for DeepSeek-V4 (dsv4_rope_attn_factor).
        var factors = RopeScaling.Yarn(g.Int("rope.dimension_count"), g.Float("attention.compress_rope_freq_base"), g.Float("rope.scaling.factor"),
            g.Int("rope.scaling.original_context_length"), g.Float("rope.scaling.yarn_beta_fast", 32f), g.Float("rope.scaling.yarn_beta_slow", 1f), 1f).Factors;
        yield return WeightInfo.Derived("rope.freq_factors_compressed", DataType.F32, new Shape(factors.Length), () =>
        {
            var bytes = new byte[factors.Length * 4];
            Buffer.BlockCopy(factors, 0, bytes, 0, bytes.Length);
            return bytes;
        });
    }
}
