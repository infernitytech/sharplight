using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face checkpoints with multi-head latent attention → canonical <c>latent_decoder</c> (vocabulary documented on
/// <c>LatentDecoderAdapter</c>): <c>deepseek_v2</c>, <c>deepseek_v3</c> / <c>kimi_k2</c>, <c>glm4_moe_lite</c> (GLM-4.7-Flash),
/// <c>minicpm3</c>, <c>hy_v4</c> (Hunyuan v4, <c>LatentDecoderHfMapping.HyV4.cs</c>) and <c>mistral4</c> (softmax routing over groups scored by their two best experts, Llama-4-style query temperature). The combined <c>kv_b_proj</c> is split into the per-head absorbed key/value up-projections; per-expert
/// tensors are stacked; multi-token-prediction layers (after <c>num_hidden_layers</c>) map like decoder layers below
/// <c>mtp.layers.{j}.</c>, with their NextN tensors (<see cref="HfNextnTensors"/>).
/// </summary>
internal sealed partial class LatentDecoderHfMapping : IHfArchitectureMapping
{
    /// <inheritdoc/>
    public string Architecture => "latent_decoder";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } =
        ["deepseek_v2", "deepseek_v3", "kimi_k2", "glm4_moe_lite", "minicpm3", "mistral4", "hy_v4", "kimi_linear", "bailing_hybrid", "deepseek_v32", "glm_moe_dsa"];

    private static bool IsMiniCpm(HfConfig c) => c.ModelType == "minicpm3";

    private static int RopeDims(HfConfig c) => c.Int("qk_rope_head_dim");

    private static int NopeDims(HfConfig c) => c.Int("qk_nope_head_dim");

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HfImportOptions options)
    {
        string a = Architecture + ".";
        int layers = c.Int("num_hidden_layers");
        int hidden = c.Int("hidden_size");
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = layers,
            [a + "hidden_size"] = hidden,
            [a + "norm_eps"] = (float)(c.GetDouble("rms_norm_eps") ?? 1e-6),
            [a + "attention.head_count"] = c.Int("num_attention_heads"),
            [a + "attention.q_rank"] = c.GetInt("q_lora_rank") ?? 0,
            [a + "attention.latent_rank"] = c.Int("kv_lora_rank"),
            [a + "attention.nope_dim"] = NopeDims(c),
            [a + "attention.rope_dim"] = RopeDims(c),
            [a + "attention.value_dim"] = c.Int("v_head_dim"),
            [a + "attention.scale"] = AttentionScale(c),
            [a + "rope.base"] = (float)(c.RopeDouble("rope_theta", "rope_theta") ?? 10000.0),
            [a + "rope.style"] = Interleaved(c) ? "gptj" : "neox",
        };
        HfMappingHelpers.AddContextLength(md, a, c);

        if (Yarn(c) is { } yarn && yarn.AttentionScale != 1f)
        {
            md[a + "rope.attention_scale"] = yarn.AttentionScale;
        }

        if (IsMiniCpm(c))
        {
            md[a + "embedding_scale"] = (float)(c.GetDouble("scale_emb") ?? 1.0);
            md[a + "residual_scale"] = (float)((c.GetDouble("scale_depth") ?? 1.0) / Math.Sqrt(layers));
            md[a + "logit_scale"] = (float)((c.GetDouble("dim_model_base") ?? hidden) / hidden);
            if (LongRope(c) is { } longRope)
            {
                md[a + "rope.original_context_length"] = longRope.OriginalContext;
                if (longRope.AttentionScale != 1f)
                {
                    md[a + "rope.attention_scale"] = longRope.AttentionScale;
                }
            }
        }

        if ((c.GetInt("n_routed_experts") ?? (IsDeltaHybrid(c) ? c.GetInt("num_experts") : null)) is { } experts)
        {
            AddMixtureOfExperts(md, a, c, experts);
        }

        if (IsDeltaHybrid(c))
        {
            AddDeltaHyperParameters(md, a, c);
        }

        AddTemperature(md, a, c);
        if (IsHyV4(c))
        {
            AddHyV4HyperParameters(md, a, c);
        }
        else if (HasIndexer(c))
        {
            md[a + "attention.indexer.token_budget"] = c.Int("index_topk");
        }

        return md;
    }

    /// <summary>
    /// DeepSeek-V3.2 and GLM-5 (DSA): a sparse-attention indexer picks <c>index_topk</c> tokens per query. Only its dense regime is implemented
    /// (the context is capped at the budget, where every visible token is kept).
    /// </summary>
    private static bool HasIndexer(HfConfig c) => c.ModelType is "deepseek_v32" or "glm_moe_dsa";

    /// <summary>DeepSeek-V2 lays the rotary dimensions out in interleaved pairs; V3-style configs say so with rope_interleave (default true).</summary>
    private static bool Interleaved(HfConfig c) => !IsMiniCpm(c) && !IsHyV4(c) && (c.GetBool("rope_interleave") ?? true);

    private static double YarnMagnitude(double factor, double mscale) => factor <= 1 ? 1.0 : 0.1 * mscale * Math.Log(factor) + 1.0;

    /// <summary>1/sqrt(nope + rope), times DeepSeek's yarn mscale(factor, mscale_all_dim)².</summary>
    private static float AttentionScale(HfConfig c)
    {
        double scale = 1.0 / Math.Sqrt(NopeDims(c) + RopeDims(c));
        if (YarnJson(c) is { } json && HfMappingHelpers.OptionalNumber(json, "mscale_all_dim") is { } allDims and not 0)
        {
            double mscale = YarnMagnitude(json.GetProperty("factor").GetDouble(), allDims);
            scale *= mscale * mscale;
        }

        return (float)scale;
    }

    private static JsonElement? YarnJson(HfConfig c)
        => c.RopeParameters is { } rp && (TypeOf(rp) == "yarn") ? rp : null;

    private static string? TypeOf(JsonElement rope)
        => rope.TryGetProperty("rope_type", out var t) && t.ValueKind == JsonValueKind.String ? t.GetString()
         : rope.TryGetProperty("type", out var t2) && t2.ValueKind == JsonValueKind.String ? t2.GetString() : null;

    /// <summary>YaRN frequency factors and the cos/sin magnitude mscale(factor, mscale) / mscale(factor, mscale_all_dim).</summary>
    private static RopeScaling.Result? Yarn(HfConfig c)
    {
        if (YarnJson(c) is not { } json)
        {
            return null;
        }

        double factor = json.GetProperty("factor").GetDouble();
        double? mscale = HfMappingHelpers.OptionalNumber(json, "mscale");
        double? allDims = HfMappingHelpers.OptionalNumber(json, "mscale_all_dim");
        double magnitude = mscale is { } m and not 0 && allDims is { } d and not 0
            ? YarnMagnitude(factor, m) / YarnMagnitude(factor, d)
            : YarnMagnitude(factor, 1.0);
        int originalContext = (int)(HfMappingHelpers.OptionalNumber(json, "original_max_position_embeddings") ?? c.GetInt("max_position_embeddings") ?? 4096);
        return RopeScaling.Yarn(
            RopeDims(c),
            (float)(c.RopeDouble("rope_theta", "rope_theta") ?? 10000.0),
            (float)factor,
            originalContext,
            (float)HfMappingHelpers.NumberOr(json, "beta_fast", 32),
            (float)HfMappingHelpers.NumberOr(json, "beta_slow", 1),
            (float)magnitude);
    }

    /// <summary>MiniCPM3 longrope: short/long per-frequency factors and the cos/sin magnitude sqrt(1 + ln(ctx/orig)/ln(orig)).</summary>
    private sealed record LongRopeFactors(float[] Short, float[] Long, int OriginalContext, float AttentionScale);

    private static LongRopeFactors? LongRope(HfConfig c)
    {
        if (c.RopeParameters is not { } rp || TypeOf(rp) is not ("longrope" or "su"))
        {
            return null;
        }

        float[] Factors(string key) => [.. rp.GetProperty(key).EnumerateArray().Select(e => (float)e.GetDouble())];
        int context = c.GetInt("max_position_embeddings") ?? 4096;
        int original = (int)(HfMappingHelpers.OptionalNumber(rp, "original_max_position_embeddings") ?? context);
        double ratio = (double)context / original;
        float magnitude = ratio <= 1 ? 1f : (float)Math.Sqrt(1 + Math.Log(ratio) / Math.Log(original));
        return new LongRopeFactors(Factors("short_factor"), Factors("long_factor"), original, magnitude);
    }

    /// <summary>Mistral 4: Llama-4-style query temperature (<c>llama_4_scaling_beta</c> over <c>original_max_position_embeddings</c>).</summary>
    /// <exception cref="InvalidDataException">The temperature beta is set without an original context length.</exception>
    private static void AddTemperature(Dictionary<string, object> md, string a, HfConfig c)
    {
        double beta = HfMappingHelpers.NumberOr(c.RopeParameters, "llama_4_scaling_beta", 0);
        if (beta == 0)
        {
            return;
        }

        double length = HfMappingHelpers.NumberOr(c.RopeParameters, "original_max_position_embeddings", 0);
        if (length <= 0)
        {
            throw new InvalidDataException($"{c.ModelType}: llama_4_scaling_beta needs rope original_max_position_embeddings.");
        }

        md[a + "attention.temperature_scale"] = (float)beta;
        md[a + "attention.temperature_length"] = (float)length;
    }

    private static void AddMixtureOfExperts(Dictionary<string, object> md, string a, HfConfig c, int experts)
    {
        string modelType = c.ModelType ?? "";
        string topkMethod = c.GetString("topk_method") ?? (modelType == "deepseek_v2" ? "greedy" : "noaux_tc");
        bool sigmoid = (c.GetString("scoring_func") ?? (modelType is "deepseek_v2" or "mistral4" ? "softmax" : "sigmoid")) == "sigmoid";
        bool normalize = c.GetBool("norm_topk_prob", "moe_renormalize") ?? false;
        double scale = c.GetDouble("routed_scaling_factor") ?? 1.0;
        md[a + "moe.expert_count"] = experts;
        md[a + "moe.expert_used_count"] = c.Int("num_experts_per_tok", "num_experts_per_token");
        md[a + "moe.router_activation"] = sigmoid ? "sigmoid" : "softmax";
        md[a + "moe.normalize_weights"] = normalize;

        // DeepSeek-V2 scales only unnormalized weights; V3-style routing normalizes and then scales.
        md[a + "moe.weights_scale"] = (float)(modelType == "deepseek_v2" && normalize ? 1.0 : scale);
        if (topkMethod is "group_limited_greedy" or "noaux_tc")
        {
            md[a + "moe.group_count"] = c.GetInt("n_group", "num_expert_group") ?? 1;
            md[a + "moe.group_used_count"] = c.GetInt("topk_group") ?? c.GetInt("n_group", "num_expert_group") ?? 1;
            md[a + "moe.group_score"] = topkMethod == "noaux_tc" ? "top2_sum" : "max";
        }
    }

    [GeneratedRegex(@"^model\.layers\.(\d+)\.(.+)$")]
    private static partial Regex LayerName();

    [GeneratedRegex(@"^mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")]
    private static partial Regex ExpertName();

    /// <summary>Per-layer tensors that are only renamed, HF suffix to canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_layer = new(StringComparer.Ordinal)
    {
        ["input_layernorm.weight"] = "input_norm.weight",
        ["post_attention_layernorm.weight"] = "post_attention_norm.weight",
        ["self_attn.q_proj.weight"] = "attention.q.weight",
        ["self_attn.q_a_proj.weight"] = "attention.q_down.weight",
        ["self_attn.q_a_layernorm.weight"] = "attention.q_down_norm.weight",
        ["self_attn.q_b_proj.weight"] = "attention.q_up.weight",
        ["self_attn.kv_a_proj_with_mqa.weight"] = "attention.kv_down.weight",
        ["self_attn.kv_a_layernorm.weight"] = "attention.kv_down_norm.weight",
        ["self_attn.o_proj.weight"] = "attention.o.weight",
        ["mlp.gate_proj.weight"] = "mlp.gate.weight",
        ["mlp.up_proj.weight"] = "mlp.up.weight",
        ["mlp.down_proj.weight"] = "mlp.down.weight",
        ["mlp.gate.weight"] = "mlp.router.weight",
        // Experts stored stacked (Mistral 4): [E, 2I, H] gate rows first, and [E, H, I].
        ["mlp.experts.gate_up_proj"] = "mlp.experts.gate_up.weight",
        ["mlp.experts.down_proj"] = "mlp.experts.down.weight",
        ["mlp.shared_experts.gate_proj.weight"] = "mlp.shared.gate.weight",
        ["mlp.shared_experts.up_proj.weight"] = "mlp.shared.up.weight",
        ["mlp.shared_experts.down_proj.weight"] = "mlp.shared.down.weight",
    };

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HfImportOptions options)
    {
        var collector = new HfWeightCollector();
        var experts = new SortedDictionary<(string Prefix, string Kind), SortedDictionary<int, WeightInfo>>();
        var convolutions = new SortedDictionary<string, WeightInfo?[]>(StringComparer.Ordinal);
        int layers = c.Int("num_hidden_layers");
        foreach (var (name, tensor) in tensors.OrderBy(p => p.Key, StringComparer.Ordinal))
        {
            var w = tensor.ToWeightInfo();
            if (TryMapGlobal(c, name, w, collector) || (IsHyV4(c) && TryMapHyV4Global(name, w, collector)))
            {
                continue;
            }

            var m = LayerName().Match(name);
            if (!m.Success)
            {
                collector.AddUnmapped(name);
                continue;
            }

            int layer = int.Parse(m.Groups[1].Value);
            string suffix = IsDeltaHybrid(c) ? NormalizeDeltaHybridSuffix(c, m.Groups[2].Value) : m.Groups[2].Value;
            // Multi-token-prediction (NextN) layers follow the trunk: an ordinary layer plus the NextN tensors, below mtp.layers.{j}.
            string prefix = layer < layers ? $"layers.{layer}." : $"mtp.layers.{layer - layers}.";
            if (layer >= layers && HfNextnTensors.Suffixes.TryGetValue(suffix, out var nextn))
            {
                if (nextn is not null)
                {
                    collector.Add(HfTransforms.Plain(w, prefix + nextn));
                }
            }
            else if (layer < layers && IsDeltaHybrid(c) && IsDeltaLayer(c, layer) && TryMapDeltaLayer(c, prefix, suffix, w, collector, convolutions))
            {
                continue;
            }
            else if (HasIndexer(c) && suffix.StartsWith("self_attn.indexer.", StringComparison.Ordinal))
            {
                continue;   // only the dense regime of the indexer is implemented: it keeps every visible token, its weights are not needed
            }
            else if (!TryMapLayer(c, prefix, suffix, w, collector, experts) && !(IsHyV4(c) && TryMapHyV4Layer(prefix, suffix, w, collector)))
            {
                collector.AddUnmapped(name);
            }
        }

        foreach (var ((prefix, kind), byIndex) in experts)
        {
            if (byIndex.Keys.Max() != byIndex.Count - 1)
            {
                throw new InvalidDataException($"{prefix}: expert indices of {kind} are not contiguous.");
            }

            collector.Add(WeightInfo.Stack($"{prefix}mlp.experts.{kind}.weight", [.. byIndex.Values]));
        }

        AddJoinedConvolutions(convolutions, collector);
        AddRopeFactors(c, collector);
        collector.ThrowIfUnmapped(c.ModelType);
        return collector.Weights;
    }

    private static bool TryMapGlobal(HfConfig c, string name, WeightInfo w, HfWeightCollector collector)
    {
        switch (name)
        {
            case "model.embed_tokens.weight" or "model.word_embeddings.weight":
                collector.Add(HfTransforms.Plain(w, "token_embedding.weight"));
                return true;
            case "model.norm.weight":
                collector.Add(HfTransforms.Plain(w, "output_norm.weight"));
                return true;
            case "lm_head.weight":
                if (!(c.GetBool("tie_word_embeddings") ?? false))
                {
                    collector.Add(HfTransforms.Plain(w, "lm_head.weight"));
                }

                return true;
            default:
                return false;
        }
    }

    private static bool TryMapLayer(
        HfConfig c, string prefix, string suffix, WeightInfo w, HfWeightCollector collector,
        SortedDictionary<(string Prefix, string Kind), SortedDictionary<int, WeightInfo>> experts)
    {
        if (ExpertName().Match(suffix) is { Success: true } expert)
        {
            var key = (prefix, expert.Groups[2].Value);
            if (!experts.TryGetValue(key, out var byIndex))
            {
                experts[key] = byIndex = [];
            }

            byIndex[int.Parse(expert.Groups[1].Value)] = w;
            return true;
        }

        if (s_layer.TryGetValue(suffix, out var renamed))
        {
            collector.Add(HfTransforms.Plain(w, prefix + renamed));
            return true;
        }

        switch (suffix)
        {
            case "self_attn.g_proj.weight":
                collector.Add(HfTransforms.Plain(w, prefix + "attention.gate.weight"));
                return true;
            case "mlp.gate.e_score_correction_bias":
                collector.Add(HfTransforms.ToF32(w, prefix + "mlp.router_bias"));
                return true;
            case "self_attn.kv_b_proj.weight":
                var (keyUp, valueUp) = LatentAttentionWeights.SplitKeyValueUp(
                    w, c.Int("num_attention_heads"), NopeDims(c), c.Int("v_head_dim"),
                    prefix + "attention.k_up.weight", prefix + "attention.v_up.weight", static t => HfTransforms.DecodeF32(t));
                collector.Add(keyUp);
                collector.Add(valueUp);
                return true;
            default:
                return false;
        }
    }

    /// <summary>Rope frequency factors, which HF derives from the config but the canonical form stores as weights.</summary>
    private static void AddRopeFactors(HfConfig c, HfWeightCollector collector)
    {
        if (Yarn(c) is { } yarn)
        {
            collector.Add(HfMappingHelpers.F32Weight("rope.freq_factors", yarn.Factors));
        }
        else if (LongRope(c) is { } longRope)
        {
            collector.Add(HfMappingHelpers.F32Weight("rope.freq_factors", longRope.Short));
            collector.Add(HfMappingHelpers.F32Weight("rope.freq_factors_long", longRope.Long));
        }
    }
}
