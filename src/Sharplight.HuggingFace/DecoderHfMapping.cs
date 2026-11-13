using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face decoder-only transformers (<c>llama</c>, <c>mistral</c>, <c>qwen2</c>, <c>qwen3</c>, <c>gemma2</c>,
/// <c>gemma3_text</c>, <c>phi3</c>, <c>olmoe</c>, <c>qwen3_moe</c>) → canonical <c>decoder</c> (vocabulary documented on <c>DecoderAdapter</c>). Weights are
/// mapped by presence (biases, q/k norms, Gemma's extra norms), Gemma's <c>1 + w</c> norms are folded, rope scaling
/// (llama3 / yarn / linear) becomes the canonical <c>rope.freq_factors</c> weight.
/// The Gemma 4 / Muse Glimmer / Flint variants live in <c>DecoderHfMapping.Extended.cs</c>.
/// </summary>
internal sealed partial class DecoderHfMapping : IHfArchitectureMapping
{
    private const double DefaultRopeBase = 10000.0;

    /// <inheritdoc/>
    public string Architecture => "decoder";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } =
    [
        "llama", "mistral", "qwen2", "qwen3", "gemma2", "gemma3_text", "phi3", "olmoe", "qwen3_moe",
        "gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text", "muse_glimmer", "flint",
    ];

    private static bool IsGemma(string? modelType) => modelType is "gemma2" or "gemma3_text" or "gemma3";

    internal static string Activation(HfConfig c)
        => (c.GetString("hidden_activation", "hidden_act", "activation_function") ?? "silu") switch
        {
            "silu" or "swish" => "silu",
            "gelu_pytorch_tanh" or "gelu_new" or "gelu_fast" or "gelu_tanh" => "gelu_tanh",
            "gelu" => "gelu",
            "relu" => "relu",
            "relu2" => "relu2",
            "xielu" => "xielu",
            var other => throw new NotSupportedException($"Hugging Face activation '{other}' is not supported."),
        };

    /// <summary>A rope scaling description: its type and the JSON object that carries the parameters.</summary>
    private sealed record Scaling(string Type, JsonElement Json);

    /// <summary>The rope scaling description of the full-attention layers, or null.</summary>
    private static Scaling? ScalingOf(HfConfig c)
    {
        if (c.RopeParameters is not { } ropeParameters)
        {
            return null;
        }

        // Newer configs may nest one rope_parameters object per layer type; the global attention one scales.
        if (ropeParameters.TryGetProperty("full_attention", out var fullAttention) && fullAttention.ValueKind == JsonValueKind.Object)
        {
            ropeParameters = fullAttention;
        }

        string type = StringProperty(ropeParameters, "rope_type") ?? StringProperty(ropeParameters, "type") ?? "default";
        return type == "default" ? null : new Scaling(type, ropeParameters);
    }

    private static string? StringProperty(JsonElement element, string key)
        => element.TryGetProperty(key, out var value) && value.ValueKind == JsonValueKind.String ? value.GetString() : null;

    internal static int RopeDim(HfConfig c, int headDim)
        => (int)(headDim * (c.RopeDouble("partial_rotary_factor", "partial_rotary_factor") ?? 1.0));

    internal static int HeadDim(HfConfig c)
        => c.GetInt("head_dim") ?? c.Int("hidden_size") / c.Int("num_attention_heads");

    internal static float RopeBase(HfConfig c) => (float)(c.RopeDouble("rope_theta", "rope_theta") ?? DefaultRopeBase);

    /// <summary>Phi-3 "longrope": short/long per-frequency factors and the cos/sin magnitude from the context extension ratio.</summary>
    private sealed record LongRopeInfo(float[] Short, float[] Long, float AttentionScale);

    private static LongRopeInfo? LongRope(HfConfig c)
    {
        if (ScalingOf(c) is not { Type: "longrope" or "su" } scaling)
        {
            return null;
        }

        float[] Factors(string key) => scaling.Json.GetProperty(key).EnumerateArray().Select(e => (float)e.GetDouble()).ToArray();
        int originalContext = c.GetInt("original_max_position_embeddings") ?? (int?)HfMappingHelpers.OptionalNumber(scaling.Json, "original_max_position_embeddings") ?? 4096;
        int maxContext = c.GetInt("max_position_embeddings") ?? originalContext;
        double scale = (double)maxContext / originalContext;
        float attentionScale = HfMappingHelpers.OptionalNumber(scaling.Json, "attention_factor") is { } explicitFactor
            ? (float)explicitFactor
            : scale <= 1 ? 1f : (float)Math.Sqrt(1 + Math.Log(scale) / Math.Log(originalContext));
        return new LongRopeInfo(Factors("short_factor"), Factors("long_factor"), attentionScale);
    }

    private static RopeScaling.Result? ComputeScaling(HfConfig c) => ScalingOf(c) is null ? null : ComputeScaling(c, RopeDim(c, HeadDim(c)), RopeBase(c));

    /// <summary>The frequency scaling of the (global-attention) <c>rope_parameters</c> for a rotary embedding of <paramref name="dim"/> dimensions and <paramref name="ropeBase"/>; null without scaling.</summary>
    internal static RopeScaling.Result? ComputeScaling(HfConfig c, int dim, float ropeBase)
    {
        if (ScalingOf(c) is not { } scaling || scaling.Type is "longrope" or "su")
        {
            return null;
        }

        var json = scaling.Json;
        float factor = (float)HfMappingHelpers.NumberOr(json, "factor", 1.0);
        int originalContext = (int)HfMappingHelpers.NumberOr(json, "original_max_position_embeddings",
            c.GetInt("original_max_position_embeddings") ?? c.GetInt("max_position_embeddings") ?? 4096);
        return scaling.Type switch
        {
            "linear" => RopeScaling.Linear(dim, factor),
            "llama3" => RopeScaling.Llama3(
                dim, ropeBase, factor,
                (float)HfMappingHelpers.NumberOr(json, "low_freq_factor", 1.0),
                (float)HfMappingHelpers.NumberOr(json, "high_freq_factor", 4.0),
                originalContext),
            "yarn" => RopeScaling.Yarn(
                dim, ropeBase, factor, originalContext,
                (float)HfMappingHelpers.NumberOr(json, "beta_fast", 32),
                (float)HfMappingHelpers.NumberOr(json, "beta_slow", 1),
                YarnAttentionFactor(json, factor),
                !json.TryGetProperty("truncate", out var truncate) || truncate.ValueKind != JsonValueKind.False),
            _ => throw new NotSupportedException($"rope scaling type '{scaling.Type}' is not supported."),
        };
    }

    /// <summary>
    /// The YaRN cos/sin magnitude as transformers computes it: <c>attention_factor</c> when given, else with both <c>mscale</c> and
    /// <c>mscale_all_dim</c> the ratio of their <c>0.1 · m · ln(factor) + 1</c> terms (DeepSeek, Ministral 3); null = the default
    /// <c>0.1 · ln(factor) + 1</c>.
    /// </summary>
    private static float? YarnAttentionFactor(JsonElement json, float factor)
    {
        if (HfMappingHelpers.OptionalNumber(json, "attention_factor") is { } explicitFactor)
        {
            return (float)explicitFactor;
        }

        double mscale = HfMappingHelpers.NumberOr(json, "mscale", 0);
        double mscaleAllDim = HfMappingHelpers.NumberOr(json, "mscale_all_dim", 0);
        if (mscale == 0 || mscaleAllDim == 0)
        {
            return null;
        }

        double Term(double m) => factor <= 1 ? 1 : 0.1 * m * Math.Log(factor) + 1;
        return (float)(Term(mscale) / Term(mscaleAllDim));
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HfImportOptions options)
    {
        string a = Architecture + ".";
        string? type = c.ModelType;
        if (type is "gemma4" or "gemma4_text" or "gemma4_unified" or "gemma4_unified_text")
        {
            return Gemma4Hyper(c);
        }

        if (type == "muse_glimmer")
        {
            return MuseHyper(c);
        }

        if (type == "flint")
        {
            return FlintHyper(c);
        }

        int layerCount = c.Int("num_hidden_layers");
        int headCount = c.Int("num_attention_heads");
        int hidden = c.Int("hidden_size");
        int headDim = HeadDim(c);
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = layerCount,
            [a + "hidden_size"] = hidden,
            [a + "ffn_hidden_size"] = c.Int("intermediate_size"),
            [a + "norm_eps"] = (float)(c.GetDouble("rms_norm_eps") ?? 1e-5),
            [a + "attention.head_count"] = headCount,
            [a + "attention.kv_head_count"] = c.GetInt("num_key_value_heads") ?? headCount,
            [a + "attention.head_dim"] = headDim,
            [a + "rope.dim"] = RopeDim(c, headDim),
            [a + "rope.base"] = RopeBase(c),
            [a + "rope.style"] = "neox",
            [a + "mlp.activation"] = Activation(c),
        };
        HfMappingHelpers.AddContextLength(md, a, c);

        if (type is "olmoe" or "qwen3_moe")
        {
            AddMixtureOfExperts(md, a, c);
        }

        AddRopeAttentionScale(md, a, c);

        if (IsGemma(type))
        {
            AddGemma(md, a, c, hidden, layerCount);
        }
        else if (type is not null && !type.StartsWith("qwen", StringComparison.Ordinal) && c.GetBool("use_sliding_window") != false
                 && c.GetInt("sliding_window") is { } slidingWindow && slidingWindow > 0)
        {
            md[a + "attention.sliding_window"] = slidingWindow;   // every layer (Mistral v0.1, Phi-3-mini-4k); null / absent = full attention
        }

        return md;
    }

    private static void AddMixtureOfExperts(Dictionary<string, object> md, string a, HfConfig c)
    {
        if (c.GetDouble("clip_qkv") is { } clip && clip > 0)
        {
            throw new NotSupportedException("clip_qkv (qkv clamping) is not supported.");
        }

        md[a + "moe.expert_count"] = c.Int("num_experts");
        md[a + "moe.expert_used_count"] = c.Int("num_experts_per_tok");
        md[a + "moe.ffn_hidden_size"] = c.GetInt("moe_intermediate_size") ?? c.Int("intermediate_size");
        md[a + "moe.normalize_weights"] = c.GetBool("norm_topk_prob") ?? false;
    }

    internal static void AddRopeAttentionScale(Dictionary<string, object> md, string a, HfConfig c)
    {
        if (c.Get("rope_scaling") is { ValueKind: JsonValueKind.Object } && LongRope(c) is { } longRope)
        {
            md[a + "rope.attention_scale"] = longRope.AttentionScale;
            md[a + "rope.original_context_length"] = c.GetInt("original_max_position_embeddings") ?? (int?)HfMappingHelpers.OptionalNumber(c.RopeParameters, "original_max_position_embeddings") ?? c.GetInt("max_position_embeddings") ?? 4096;
        }
        else if (ComputeScaling(c) is { } scaling && scaling.AttentionScale != 1f)
        {
            md[a + "rope.attention_scale"] = scaling.AttentionScale;
        }
    }

    private static void AddGemma(Dictionary<string, object> md, string a, HfConfig c, int hidden, int layerCount)
    {
        string? type = c.ModelType;
        md[a + "embedding_scale"] = MathF.Sqrt(hidden);
        if (c.GetDouble("query_pre_attn_scalar") is { } queryScalar)
        {
            md[a + "attention.scale"] = (float)(1.0 / Math.Sqrt(queryScalar));
        }

        if (c.GetDouble("attn_logit_softcapping") is { } attentionSoftcap && attentionSoftcap > 0)
        {
            md[a + "attention.logit_softcap"] = (float)attentionSoftcap;
        }

        if (c.GetDouble("final_logit_softcapping") is { } finalSoftcap && finalSoftcap > 0)
        {
            md[a + "final_logit_softcap"] = (float)finalSoftcap;
        }

        if (c.GetInt("sliding_window") is not { } window || window <= 0)
        {
            return;
        }

        md[a + "attention.sliding_window"] = window;
        md[a + "attention.sliding_layers"] = GemmaSlidingLayers(c, type, layerCount);
        if (c.GetDouble("rope_local_base_freq") is { } localBase)
        {
            md[a + "rope.local_base"] = (float)localBase;
        }
        else if (type == "gemma3_text")
        {
            md[a + "rope.local_base"] = (float)DefaultRopeBase;
        }
    }

    private static int[] GemmaSlidingLayers(HfConfig c, string? type, int layerCount)
    {
        if (c.GetStringArray("layer_types") is { } layerTypes)
        {
            return HfMappingHelpers.SlidingFlags(layerTypes);
        }

        // gemma2: even layers slide; gemma3: every sliding_window_pattern-th layer is global.
        int pattern = type == "gemma2" ? 2 : c.GetInt("sliding_window_pattern") ?? 6;
        return Enumerable.Range(0, layerCount).Select(il => il % pattern < pattern - 1 ? 1 : 0).ToArray();
    }

    // ---- weights ------------------------------------------------------------------------------
    [GeneratedRegex(@"^model\.layers\.(\d+)\.(.+)$")]
    private static partial Regex LayerName();

    [GeneratedRegex(@"^mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")]
    private static partial Regex ExpertName();

    /// <summary>Per-layer tensors that are only renamed, HF suffix to canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_layer = new(StringComparer.Ordinal)
    {
        ["self_attn.q_proj.weight"] = "attention.q.weight",
        ["self_attn.k_proj.weight"] = "attention.k.weight",
        ["self_attn.v_proj.weight"] = "attention.v.weight",
        ["self_attn.o_proj.weight"] = "attention.o.weight",
        ["self_attn.q_proj.bias"] = "attention.q.bias",
        ["self_attn.k_proj.bias"] = "attention.k.bias",
        ["self_attn.v_proj.bias"] = "attention.v.bias",
        ["self_attn.o_proj.bias"] = "attention.o.bias",
        ["mlp.gate_proj.weight"] = "mlp.gate.weight",
        ["mlp.up_proj.weight"] = "mlp.up.weight",
        ["mlp.down_proj.weight"] = "mlp.down.weight",
        ["mlp.gate_proj.bias"] = "mlp.gate.bias",
        ["mlp.up_proj.bias"] = "mlp.up.bias",
        ["mlp.down_proj.bias"] = "mlp.down.bias",
        // Flint: attention sinks and the always-on shared expert.
        ["self_attn.sinks"] = "attention.sinks",
        ["self_attn.softmax_offset"] = "attention.sinks",   // the name transformers 5 writes
        ["mlp.shared_expert.gate_proj.weight"] = "mlp.shared.gate.weight",
        ["mlp.shared_expert.up_proj.weight"] = "mlp.shared.up.weight",
        ["mlp.shared_expert.down_proj.weight"] = "mlp.shared.down.weight",
    };

    /// <summary>Norm tensors by HF suffix; they go through the model's norm convention (<c>1 + w</c> folding for Gemma).</summary>
    private static readonly Dictionary<string, string> s_norms = new(StringComparer.Ordinal)
    {
        ["input_layernorm.weight"] = "input_norm.weight",
        ["self_attn.q_norm.weight"] = "attention.q_norm.weight",
        ["self_attn.k_norm.weight"] = "attention.k_norm.weight",
    };

    /// <summary>How the norm tensors of one checkpoint are interpreted (see <see cref="DecoderHfMapping.MapWeights"/>).</summary>
    private sealed record WeightContext(bool Extended, bool Gemma, bool PlusOne, string PostAttentionNorm)
    {
        public WeightInfo Norm(WeightInfo w, string name) => PlusOne ? HfTransforms.NormPlusOne(w, name) : HfTransforms.Plain(w, name);
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HfImportOptions options)
    {
        bool extended = IsGemma4OrMuse(c.ModelType);
        bool gemma = IsGemma(c.ModelType) || extended;
        // Gemma 2/3 and Muse layer norms store w - 1 (Gemma 4's store w).
        bool plusOne = IsGemma(c.ModelType) || c.ModelType == "muse_glimmer";
        // Gemma: post_attention_layernorm is the attention output norm, pre_feedforward_layernorm precedes the MLP.
        var context = new WeightContext(extended, gemma, plusOne, gemma ? "attention.out_norm.weight" : "post_attention_norm.weight");

        var collector = new HfWeightCollector();
        var experts = new SortedDictionary<(int Layer, string Kind), SortedDictionary<int, WeightInfo>>();

        // Base-model checkpoints (e.g. Qwen3-Embedding saved as Qwen3Model) store the trunk without the "model." prefix.
        bool bare = !tensors.Keys.Any(k => k.StartsWith("model.", StringComparison.Ordinal)) && tensors.ContainsKey("embed_tokens.weight");
        foreach (var (rawName, tensor) in tensors.OrderBy(p => p.Key, StringComparer.Ordinal))
        {
            string name = bare && !rawName.StartsWith("lm_head.", StringComparison.Ordinal) ? "model." + rawName : rawName;
            if (name.EndsWith("rotary_emb.inv_freq", StringComparison.Ordinal))
            {
                continue;
            }

            var w = tensor.ToWeightInfo();
            if (extended)
            {
                if (IsNonTextTensor(name))
                {
                    continue;
                }

                name = name.Replace("model.language_model.", "model.");
            }

            if (TryMapGlobalTensor(c, context, name, w, collector))
            {
                continue;
            }

            var m = LayerName().Match(name);
            if (!m.Success || !TryMapLayerTensor(c, context, m.Groups[1].Value, m.Groups[2].Value, w, collector, experts))
            {
                collector.AddUnmapped(name);
            }
        }

        StackExperts(c, experts, collector);
        collector.ThrowIfUnmapped(c.ModelType);
        AddRopeFactors(c, extended, collector);
        return collector.Weights;
    }

    /// <summary>Embedding, final norm and output head; false when <paramref name="name"/> is none of them.</summary>
    private static bool TryMapGlobalTensor(HfConfig c, WeightContext context, string name, WeightInfo w, HfWeightCollector collector)
    {
        switch (name)
        {
            case "model.embed_tokens.weight":
                collector.Add(HfTransforms.Plain(w, "token_embedding.weight"));
                return true;
            case "model.embed_tokens_per_layer.weight" when context.Extended:
                collector.Add(HfTransforms.Plain(w, "per_layer.embedding.weight"));
                return true;
            case "model.per_layer_model_projection.weight" when context.Extended:
                collector.Add(HfTransforms.Plain(w, "per_layer.model_projection.weight"));
                return true;
            case "model.per_layer_projection_norm.weight" when context.Extended:
                collector.Add(HfTransforms.Plain(w, "per_layer.projection_norm.weight"));
                return true;
            case "model.norm.weight":
                collector.Add(c.ModelType == "muse_glimmer" ? HfTransforms.Plain(w, "output_norm.weight") : context.Norm(w, "output_norm.weight"));
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

    /// <summary>One <c>model.layers.N.*</c> tensor; false when it has no canonical counterpart.</summary>
    private static bool TryMapLayerTensor(
        HfConfig c,
        WeightContext context,
        string layer,
        string suffix,
        WeightInfo w,
        HfWeightCollector collector,
        SortedDictionary<(int Layer, string Kind), SortedDictionary<int, WeightInfo>> experts)
    {
        string prefix = $"layers.{layer}.";
        if (ExpertName().Match(suffix) is { Success: true } expertMatch)
        {
            // Per-expert tensors are stacked into one weight per layer and kind once all are seen.
            var key = (int.Parse(layer), expertMatch.Groups[2].Value);
            if (!experts.TryGetValue(key, out var byIndex))
            {
                experts[key] = byIndex = [];
            }

            byIndex[int.Parse(expertMatch.Groups[1].Value)] = w;
            return true;
        }

        if (suffix == "mlp.gate.weight" && c.Has("num_experts"))
        {
            collector.Add(HfTransforms.Plain(w, prefix + "mlp.router.weight"));
        }
        else if (s_layer.TryGetValue(suffix, out var renamed))
        {
            collector.Add(HfTransforms.Plain(w, prefix + renamed));
        }
        else if (s_norms.TryGetValue(suffix, out var normName))
        {
            collector.Add(context.Norm(w, prefix + normName));
        }
        else if (suffix == "post_attention_layernorm.weight")
        {
            collector.Add(context.Norm(w, prefix + context.PostAttentionNorm));
        }
        else if (context.Gemma && suffix == "pre_feedforward_layernorm.weight")
        {
            collector.Add(context.Norm(w, prefix + "post_attention_norm.weight"));
        }
        else if (suffix == "self_attn.qkv_proj.weight")
        {
            SplitFusedAttention(c, prefix, w, collector);
        }
        else if (suffix == "mlp.gate_up_proj.weight")
        {
            int ffnLength = c.Int("intermediate_size");
            collector.Add(HfTransforms.Plain(w.RowSlice("", 0, ffnLength), prefix + "mlp.gate.weight"));
            collector.Add(HfTransforms.Plain(w.RowSlice("", ffnLength, ffnLength), prefix + "mlp.up.weight"));
        }
        else if (context.Gemma && suffix == "post_feedforward_layernorm.weight")
        {
            collector.Add(context.Norm(w, prefix + "mlp.out_norm.weight"));
        }
        else if (context.Extended && s_extended.TryGetValue(suffix, out var extendedName))
        {
            collector.Add(extendedName.EndsWith("norm.weight") ? context.Norm(w, prefix + extendedName) : HfTransforms.Plain(w, prefix + extendedName));
        }
        else
        {
            return false;
        }

        return true;
    }

    /// <summary>Phi-3 stores q/k/v as one tensor (q rows first).</summary>
    private static void SplitFusedAttention(HfConfig c, string prefix, WeightInfo w, HfWeightCollector collector)
    {
        int headCount = c.Int("num_attention_heads");
        int kvHeadCount = c.GetInt("num_key_value_heads") ?? headCount;
        int headDim = HeadDim(c);
        collector.Add(HfTransforms.Plain(w.RowSlice("", 0, headCount * headDim), prefix + "attention.q.weight"));
        collector.Add(HfTransforms.Plain(w.RowSlice("", headCount * headDim, kvHeadCount * headDim), prefix + "attention.k.weight"));
        collector.Add(HfTransforms.Plain(w.RowSlice("", (headCount + kvHeadCount) * headDim, kvHeadCount * headDim), prefix + "attention.v.weight"));
    }

    private static void StackExperts(
        HfConfig c,
        SortedDictionary<(int Layer, string Kind), SortedDictionary<int, WeightInfo>> experts,
        HfWeightCollector collector)
    {
        foreach (var ((layer, kind), byIndex) in experts)
        {
            if (byIndex.Count != c.Int("num_experts") || byIndex.Keys.Max() != byIndex.Count - 1)
            {
                throw new InvalidDataException($"layer {layer}: expected {c.Int("num_experts")} {kind} experts, found {byIndex.Count}.");
            }

            collector.Add(WeightInfo.Stack($"layers.{layer}.mlp.experts.{kind}.weight", [.. byIndex.Values]));
        }
    }

    /// <summary>Rope frequency factors, which HF derives from the config but the canonical form stores as weights.</summary>
    internal static void AddRopeFactors(HfConfig c, bool extended, HfWeightCollector collector)
    {
        if (extended)
        {
            if (ExtendedRopeFactors(c) is { } factors)
            {
                collector.Add(HfMappingHelpers.F32Weight("rope.freq_factors", factors));
            }

            if (c.ModelType == "muse_glimmer")
            {
                foreach (var norm in MuseQkNorms(c))
                {
                    collector.Add(norm);
                }
            }
        }
        else if (LongRope(c) is { } longRope)
        {
            collector.Add(HfMappingHelpers.F32Weight("rope.freq_factors", longRope.Short));
            collector.Add(HfMappingHelpers.F32Weight("rope.freq_factors_long", longRope.Long));
        }
        else if (ComputeScaling(c) is { } scaling)
        {
            collector.Add(HfMappingHelpers.F32Weight("rope.freq_factors", scaling.Factors));
        }
    }
}
