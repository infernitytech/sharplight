using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp architectures with multi-head latent attention → canonical <c>latent_decoder</c> (vocabulary documented on
/// <c>LatentDecoderAdapter</c>): <c>deepseek2</c> (DeepSeek-V2/V2-Lite/V3/R1, Kimi-K2, GLM-4.7-Flash, Moonlight) and
/// <c>minicpm3</c>, and <c>mistral4</c> (Mistral Small 4: softmax routing with group scores of the two best experts, fused gate+up experts, query
/// temperature, which Mistral Large 3 <c>deepseek2</c> files carry too) and <c>plm</c> (PLM-1.8B: direct query projection, ungated squared-ReLU MLP,
/// output head tied to the embedding). Files written by older converters store the key/value up-projection as one <c>attn_kv_b</c> tensor, which is
/// split (and its key half transposed per head) into the canonical absorbed form; newer files carry <c>attn_k_b</c> /
/// <c>attn_v_b</c> directly. <c>deepseek32</c> and <c>glm-dsa</c> (DeepSeek-V3.2, GLM-5: the DSA indexer, <c>attention.indexer.top_k</c>). <c>kimi-linear</c> and <c>bailingmoe3</c> add delta-attention layers (see the DeltaAttention part),
/// <c>glm5-next</c> (GLM-5.3-Flash) also hyper-connections (see the Glm5Next part).
/// </summary>
internal sealed partial class LatentDecoderGgufMapping : IGgufArchitectureMapping, IGgufSplitMapping
{
    // MiniCPM3 constants llama.cpp hard-codes because its GGUFs do not store them (HF config: scale_emb, scale_depth, dim_model_base).
    private const float MiniCpmEmbeddingScale = 12f;
    private const float MiniCpmDepthScale = 1.4f;
    private const float MiniCpmBaseWidth = 256f;

    // llama.cpp expert_gating_func values.
    private const int GatingSigmoid = 2;

    /// <inheritdoc/>
    public IReadOnlyList<string> GgufArchitectures { get; } = ["deepseek2", "minicpm3", "mistral4", "plm", "kimi-linear", "bailingmoe3", "deepseek32", "glm-dsa", "glm5-next"];

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

        if (g.GgufArchitecture == "plm")
        {
            md[a + "mlp.activation"] = "relu2";
            md[a + "mlp.gated"] = false;
        }

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

        if (IsDeltaHybrid(g))
        {
            AddDeltaHyperParameters(md, a, g);
        }

        if (IsGlm5Next(g))
        {
            AddGlm5NextHyperParameters(md, a, g);
        }
        else if (g.Has("attention.indexer.top_k"))
        {
            // DSA (deepseek32, glm-dsa): llama.cpp rotates the indexer's rope dimensions with NEOX pairs for DeepSeek-V3.2 and NORM pairs for GLM-5.
            md[a + "attention.indexer.token_budget"] = g.Int("attention.indexer.top_k");
            md[a + "attention.indexer.rope_style"] = g.GgufArchitecture == "glm-dsa" ? "gptj" : "neox";
        }

        // Mistral 4 / Mistral Large 3: Llama-4-style query temperature beyond the original context.
        if (g.Float("attention.temperature_scale", 0f) is var temperature && temperature != 0f)
        {
            int length = g.Int("attention.temperature_length", g.Int("rope.scaling.original_context_length", 0));
            if (length <= 0)
            {
                throw new InvalidDataException($"{g.GgufArchitecture}: attention.temperature_scale needs attention.temperature_length.");
            }

            md[a + "attention.temperature_scale"] = temperature;
            md[a + "attention.temperature_length"] = (float)length;
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
        // Mistral 4 scores its groups by the two best softmax scores as llama.cpp does; DeepSeek-V2's softmax groups use their best score.
        md[a + "moe.group_score"] = sigmoid || g.GgufArchitecture == "mistral4" ? "top2_sum" : "max";
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
        ["ffn_gate_up_exps.weight"] = "mlp.experts.gate_up.weight",
        ["ffn_gate_shexp.weight"] = "mlp.shared.gate.weight",
        ["ffn_up_shexp.weight"] = "mlp.shared.up.weight",
        ["ffn_down_shexp.weight"] = "mlp.shared.down.weight",
        // DSA indexer.
        ["indexer.attn_q_b.weight"] = "attention.indexer.q_up.weight",
        ["indexer.attn_k.weight"] = "attention.indexer.k.weight",
        ["indexer.k_norm.weight"] = "attention.indexer.k_norm.weight",
        ["indexer.k_norm.bias"] = "attention.indexer.k_norm.bias",
        ["indexer.proj.weight"] = "attention.indexer.head_weights.weight",
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
            // Multi-token-prediction (NextN) layers follow the trunk: ordinary layers plus the NextN tensors, below mtp.layers.{j}.
            string mtpPrefix = $"mtp.layers.{layer - TrunkLayers(g)}.";
            return GgufDecoderFamilyMapping.NextnTensors.TryGetValue(suffix, out var nextn) || s_layer.TryGetValue(suffix, out nextn)
                ? mtpPrefix + nextn
                : mtpPrefix + suffix;   // unknown to the head: kept under its GGUF name
        }

        if (IsDeltaHybrid(g) && IsDeltaLayer(g, layer) && MapDeltaSuffix(g, suffix) is { } delta)
        {
            return $"layers.{layer}.{delta}";
        }

        if (IsGlm5Next(g) && s_hyperConnections.TryGetValue(suffix, out var hyperConnection))
        {
            return $"layers.{layer}.{hyperConnection}";
        }

        if (IsBailing3(g) && suffix == "attn_gate.weight")
        {
            return $"layers.{layer}.attention.gate.weight";   // one sigmoid gate per head
        }

        return s_layer.TryGetValue(suffix, out var canonical) ? $"layers.{layer}.{canonical}" : null;
    }

    /// <inheritdoc/>
    public WeightInfo MapWeight(WeightInfo weight, GgufMetadataView g) => IsDeltaHybrid(g) ? DeltaVector(weight, g) ?? weight : weight;

    /// <summary>
    /// Older files: the combined key/value up-projection splits into the per-head key (transposed) and value projections. GLM-5.3-Flash's k-pool
    /// indexer tensors split into nothing (dropped).
    /// </summary>
    public IReadOnlyList<WeightInfo>? Split(string name, WeightInfo t, GgufMetadataView g)
    {
        var m = GgufTensorNames.Block().Match(name);
        if (m.Success && IsGlm5Next(g) && IsPoolIndexerTensor(m.Groups[2].Value))
        {
            return [];   // the k-pool indexer is not used (dense attention up to its budget)
        }

        if (!m.Success || m.Groups[2].Value != "attn_kv_b.weight")
        {
            return null;
        }

        int layer = int.Parse(m.Groups[1].Value);
        string prefix = layer < TrunkLayers(g) ? $"layers.{layer}." : $"mtp.layers.{layer - TrunkLayers(g)}.";
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
        foreach (var joined in JoinedConvolutions(weights))
        {
            yield return joined;
        }

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
