using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp architectures with multi-head latent attention → canonical <c>latent_decoder</c> (vocabulary documented on
/// <c>LatentDecoderAdapter</c>): <c>deepseek2</c> (DeepSeek-V2/V2-Lite/V3/R1, Kimi-K2, GLM-4.7-Flash, Moonlight) and
/// <c>minicpm3</c>. Files written by older converters store the key/value up-projection as one <c>attn_kv_b</c> tensor, which is
/// split (and its key half transposed per head) into the canonical absorbed form; newer files carry <c>attn_k_b</c> /
/// <c>attn_v_b</c> directly.
/// </summary>
internal sealed class LatentDecoderGgufMapping : IGgufArchitectureMapping, IGgufSplitMapping
{
    // MiniCPM3 constants llama.cpp hard-codes because its GGUFs do not store them (HF config: scale_emb, scale_depth, dim_model_base).
    private const float MiniCpmEmbeddingScale = 12f;
    private const float MiniCpmDepthScale = 1.4f;
    private const float MiniCpmBaseWidth = 256f;

    // llama.cpp expert_gating_func values.
    private const int GatingSigmoid = 2;

    /// <inheritdoc/>
    public IReadOnlyList<string> GgufArchitectures { get; } = ["deepseek2", "minicpm3"];

    /// <inheritdoc/>
    public string Architecture => "latent_decoder";

    private static bool IsMiniCpm(GgufMetadataView g) => g.GgufArchitecture == "minicpm3";

    /// <summary>Query/key head size (non-rotary + rotary); newer converters store the absorbed sizes as key_length and the real ones as key_length_mla.</summary>
    private static int QueryKeyDims(GgufMetadataView g) => g.Int("attention.key_length_mla", g.Int("attention.key_length"));

    private static int ValueDims(GgufMetadataView g)
        => g.Int("attention.value_length_mla", g.Int("attention.value_length", g.Int("embedding_length") / g.Int("attention.head_count")));

    private static int TrunkLayers(GgufMetadataView g) => g.Int("block_count") - g.Int("nextn_predict_layers", 0);

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(GgufMetadataView g)
    {
        string a = Architecture + ".";
        int ropeDims = g.Int("rope.dimension_count");
        int layers = TrunkLayers(g);
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = layers,
            [a + "hidden_size"] = g.Int("embedding_length"),
            [a + "norm_eps"] = g.Float("attention.layer_norm_rms_epsilon"),
            [a + "attention.head_count"] = g.Int("attention.head_count"),
            [a + "attention.q_rank"] = g.Int("attention.q_lora_rank", 0),
            [a + "attention.latent_rank"] = g.Int("attention.kv_lora_rank"),
            [a + "attention.nope_dim"] = QueryKeyDims(g) - ropeDims,
            [a + "attention.rope_dim"] = ropeDims,
            [a + "attention.value_dim"] = ValueDims(g),
            [a + "attention.scale"] = AttentionScale(g),
            [a + "rope.base"] = g.Float("rope.freq_base", GgufMappingHelpers.DefaultRopeBase),
            [a + "rope.style"] = IsMiniCpm(g) ? "neox" : "gptj",
        };
        GgufMappingHelpers.AddContextLength(md, a, g);

        if (IsMiniCpm(g))
        {
            md[a + "embedding_scale"] = MiniCpmEmbeddingScale;
            md[a + "residual_scale"] = MiniCpmDepthScale / MathF.Sqrt(layers);
            md[a + "logit_scale"] = MiniCpmBaseWidth / g.Int("embedding_length");
            md[a + "rope.original_context_length"] = g.Int("rope.scaling.original_context_length", g.Int("context_length"));
        }

        if (g.Has("expert_count"))
        {
            AddMixtureOfExperts(md, a, g);
        }

        return md;
    }

    /// <summary>1/sqrt(query/key head size), times yarn's squared magnitude correction (DeepSeek folds it into the softmax scale).</summary>
    private static float AttentionScale(GgufMetadataView g)
    {
        float scale = 1f / MathF.Sqrt(QueryKeyDims(g));
        if (g.Raw("rope.scaling.type") is "yarn" && g.Has("rope.scaling.yarn_log_multiplier"))
        {
            float mscale = 1f + g.Float("rope.scaling.yarn_log_multiplier") * MathF.Log(g.Float("rope.scaling.factor"));
            scale *= mscale * mscale;
        }

        return scale;
    }

    private static void AddMixtureOfExperts(Dictionary<string, object> md, string a, GgufMetadataView g)
    {
        bool sigmoid = g.Int("expert_gating_func", 1) == GatingSigmoid;
        md[a + "moe.expert_count"] = g.Int("expert_count");
        md[a + "moe.expert_used_count"] = g.Int("expert_used_count");
        md[a + "moe.router_activation"] = sigmoid ? "sigmoid" : "softmax";
        md[a + "moe.normalize_weights"] = g.Bool("expert_weights_norm", false);
        md[a + "moe.weights_scale"] = g.Float("expert_weights_scale", 1f);
        md[a + "moe.group_count"] = g.Int("expert_group_count", 1);
        md[a + "moe.group_used_count"] = g.Int("expert_group_used_count", g.Int("expert_group_count", 1));
        md[a + "moe.group_score"] = sigmoid ? "top2_sum" : "max";
    }

    /// <summary>Per-layer tensors, GGUF suffix to canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_layer = new(StringComparer.Ordinal)
    {
        ["attn_norm.weight"] = "input_norm.weight",
        ["ffn_norm.weight"] = "post_attention_norm.weight",
        ["attn_q.weight"] = "attention.q.weight",
        ["attn_q_a.weight"] = "attention.q_down.weight",
        ["attn_q_a_norm.weight"] = "attention.q_down_norm.weight",
        ["attn_q_b.weight"] = "attention.q_up.weight",
        ["attn_kv_a_mqa.weight"] = "attention.kv_down.weight",
        ["attn_kv_a_norm.weight"] = "attention.kv_down_norm.weight",
        ["attn_k_b.weight"] = "attention.k_up.weight",
        ["attn_v_b.weight"] = "attention.v_up.weight",
        ["attn_output.weight"] = "attention.o.weight",
        ["ffn_gate.weight"] = "mlp.gate.weight",
        ["ffn_up.weight"] = "mlp.up.weight",
        ["ffn_down.weight"] = "mlp.down.weight",
        ["ffn_gate_inp.weight"] = "mlp.router.weight",
        ["exp_probs_b.bias"] = "mlp.router_bias",
        ["ffn_gate_exps.weight"] = "mlp.experts.gate.weight",
        ["ffn_up_exps.weight"] = "mlp.experts.up.weight",
        ["ffn_down_exps.weight"] = "mlp.experts.down.weight",
        ["ffn_gate_shexp.weight"] = "mlp.shared.gate.weight",
        ["ffn_up_shexp.weight"] = "mlp.shared.up.weight",
        ["ffn_down_shexp.weight"] = "mlp.shared.down.weight",
    };

    /// <summary>Rope frequency tensors outside the layers.</summary>
    private static readonly Dictionary<string, string> s_ropeFactors = new(StringComparer.Ordinal)
    {
        ["rope_freqs.weight"] = "rope.freq_factors",
        ["rope_factors_short.weight"] = "rope.freq_factors",
        ["rope_factors_long.weight"] = "rope.freq_factors_long",
    };

    /// <inheritdoc/>
    public string? MapWeightName(string name, GgufMetadataView g)
    {
        if (GgufTensorNames.MapGlobal(name) is { } globalName)
        {
            return globalName;
        }

        if (s_ropeFactors.TryGetValue(name, out var ropeName))
        {
            return ropeName;
        }

        var m = GgufTensorNames.Block().Match(name);
        if (!m.Success)
        {
            return null;
        }

        int layer = int.Parse(m.Groups[1].Value);
        string suffix = m.Groups[2].Value;
        if (layer >= TrunkLayers(g))
        {
            // Multi-token-prediction (NextN) layers follow the trunk; their tensors stay under mtp.* and are not part of the decode graph.
            string mtpName = suffix.StartsWith("nextn.", StringComparison.Ordinal) || suffix == "attn_kv_b.weight" ? suffix : s_layer.GetValueOrDefault(suffix) ?? suffix;
            return $"mtp.layers.{layer - TrunkLayers(g)}.{mtpName}";
        }

        return s_layer.TryGetValue(suffix, out var canonical) ? $"layers.{layer}.{canonical}" : null;
    }

    /// <summary>Older files: the combined key/value up-projection splits into the per-head key (transposed) and value projections.</summary>
    public IReadOnlyList<WeightInfo>? Split(string name, WeightInfo t, GgufMetadataView g)
    {
        var m = GgufTensorNames.Block().Match(name);
        if (!m.Success || m.Groups[2].Value != "attn_kv_b.weight" || int.Parse(m.Groups[1].Value) >= TrunkLayers(g))
        {
            return null;
        }

        string prefix = $"layers.{m.Groups[1].Value}.";
        int ropeDims = g.Int("rope.dimension_count");
        var (keyUp, valueUp) = LatentAttentionWeights.SplitKeyValueUp(
            t, g.Int("attention.head_count"), QueryKeyDims(g) - ropeDims, ValueDims(g),
            prefix + "attention.k_up.weight", prefix + "attention.v_up.weight",
            static w => GgmlBlockDecoder.Decode(w.Type, w.ReadAll()));
        return [keyUp, valueUp];
    }

    /// <inheritdoc/>
    public IEnumerable<WeightInfo> SyntheticWeights(GgufMetadataView g, IReadOnlyDictionary<string, WeightInfo> weights)
    {
        if (weights.ContainsKey("rope.freq_factors") || g.Raw("rope.scaling.type") is not "yarn")
        {
            yield break;
        }

        var yarn = RopeScaling.Yarn(
            g.Int("rope.dimension_count"),
            g.Float("rope.freq_base", GgufMappingHelpers.DefaultRopeBase),
            g.Float("rope.scaling.factor"),
            g.Int("rope.scaling.original_context_length"));
        var factors = yarn.Factors;
        yield return WeightInfo.Derived("rope.freq_factors", DataType.F32, new Shape(factors.Length), () =>
        {
            var bytes = new byte[factors.Length * 4];
            Buffer.BlockCopy(factors, 0, bytes, 0, bytes.Length);
            return bytes;
        });
    }
}
