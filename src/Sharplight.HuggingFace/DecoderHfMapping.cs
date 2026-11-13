using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face decoder-only transformers (<c>llama</c>, <c>mistral</c>, <c>qwen2</c>, <c>qwen3</c>, <c>gemma2</c>,
/// <c>gemma3_text</c>, <c>phi3</c>, <c>olmoe</c>, <c>qwen3_moe</c>) → canonical <c>decoder</c> (vocabulary documented on <c>DecoderAdapter</c>). Weights are
/// mapped by presence (biases, q/k norms, Gemma's extra norms), Gemma's <c>1 + w</c> norms are folded, rope scaling
/// (llama3 / yarn / linear) becomes the canonical <c>rope.freq_factors</c> weight.
/// </summary>
public sealed partial class DecoderHfMapping : IHfArchitectureMapping
{
    public string Architecture => "decoder";
    public IReadOnlyList<string> ModelTypes { get; } = ["llama", "mistral", "qwen2", "qwen3", "gemma2", "gemma3_text", "phi3", "olmoe", "qwen3_moe", "gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text", "muse_glimmer", "flint"];

    static bool IsGemma(string? t) => t is "gemma2" or "gemma3_text" or "gemma3";

    static string Activation(HfConfig c)
        => (c.GetString("hidden_activation", "hidden_act", "activation_function") ?? "silu") switch
        {
            "silu" or "swish" => "silu",
            "gelu_pytorch_tanh" or "gelu_new" or "gelu_fast" or "gelu_tanh" => "gelu_tanh",
            "gelu" => "gelu",
            "relu" => "relu",
            var other => throw new NotSupportedException($"Hugging Face activation '{other}' is not supported."),
        };

    sealed record Scaling(string Type, JsonElement Json);

    /// <summary>The rope scaling description of the full-attention layers, or null.</summary>
    static Scaling? ScalingOf(HfConfig c)
    {
        if (c.RopeParameters is not { } rp) return null;
        // Newer configs may nest one rope_parameters object per layer type; the global attention one scales.
        if (rp.TryGetProperty("full_attention", out var fa) && fa.ValueKind == JsonValueKind.Object) rp = fa;
        string type = rp.TryGetProperty("rope_type", out var rt) && rt.ValueKind == JsonValueKind.String ? rt.GetString()!
                    : rp.TryGetProperty("type", out var t) && t.ValueKind == JsonValueKind.String ? t.GetString()! : "default";
        return type == "default" ? null : new Scaling(type, rp);
    }

    static double D(JsonElement e, string key, double def)
        => e.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetDouble() : def;

    static int RopeDim(HfConfig c, int headDim)
        => (int)(headDim * (c.RopeDouble("partial_rotary_factor", "partial_rotary_factor") ?? 1.0));

    static int HeadDim(HfConfig c)
        => c.GetInt("head_dim") ?? c.Int("hidden_size") / c.Int("num_attention_heads");

    static float RopeBase(HfConfig c) => (float)(c.RopeDouble("rope_theta", "rope_theta") ?? 10000.0);

    /// <summary>Phi-3 "longrope": short/long per-frequency factors and the cos/sin magnitude from the context extension ratio.</summary>
    sealed record LongRopeInfo(float[] Short, float[] Long, float AttentionScale);

    static LongRopeInfo? LongRope(HfConfig c)
    {
        if (ScalingOf(c) is not { Type: "longrope" or "su" } s) return null;
        float[] Arr(string key) => s.Json.GetProperty(key).EnumerateArray().Select(e => (float)e.GetDouble()).ToArray();
        int orig = c.GetInt("original_max_position_embeddings") ?? 4096, max = c.GetInt("max_position_embeddings") ?? orig;
        double scale = (double)max / orig;
        float attn = s.Json.TryGetProperty("attention_factor", out var af) && af.ValueKind == JsonValueKind.Number ? (float)af.GetDouble()
                   : scale <= 1 ? 1f : (float)Math.Sqrt(1 + Math.Log(scale) / Math.Log(orig));
        return new LongRopeInfo(Arr("short_factor"), Arr("long_factor"), attn);
    }

    static RopeScaling.Result? ComputeScaling(HfConfig c)
    {
        if (ScalingOf(c) is not { } s) return null;
        if (s.Type is "longrope" or "su") return null;
        int dim = RopeDim(c, HeadDim(c));
        float @base = RopeBase(c);
        var j = s.Json;
        float factor = (float)D(j, "factor", 1.0);
        int orig = (int)D(j, "original_max_position_embeddings", c.GetInt("original_max_position_embeddings") ?? c.GetInt("max_position_embeddings") ?? 4096);
        return s.Type switch
        {
            "linear" => RopeScaling.Linear(dim, factor),
            "llama3" => RopeScaling.Llama3(dim, @base, factor, (float)D(j, "low_freq_factor", 1.0), (float)D(j, "high_freq_factor", 4.0), orig),
            "yarn" => RopeScaling.Yarn(dim, @base, factor, orig, (float)D(j, "beta_fast", 32), (float)D(j, "beta_slow", 1),
                                       j.TryGetProperty("attention_factor", out var af) && af.ValueKind == JsonValueKind.Number ? (float)af.GetDouble() : null,
                                       !j.TryGetProperty("truncate", out var tr) || tr.ValueKind != JsonValueKind.False),
            _ => throw new NotSupportedException($"rope scaling type '{s.Type}' is not supported."),
        };
    }

    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HfImportOptions options)
    {
        string a = Architecture + ".";
        string? type = c.ModelType;
        if (type is "gemma4" or "gemma4_text" or "gemma4_unified" or "gemma4_unified_text") return Gemma4Hyper(c);
        if (type == "muse_glimmer") return MuseHyper(c);
        if (type == "flint") return FlintHyper(c);
        int nLayer = c.Int("num_hidden_layers");
        int nHead = c.Int("num_attention_heads");
        int hidden = c.Int("hidden_size");
        int headDim = HeadDim(c);
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = nLayer,
            [a + "hidden_size"] = hidden,
            [a + "ffn_hidden_size"] = c.Int("intermediate_size"),
            [a + "norm_eps"] = (float)(c.GetDouble("rms_norm_eps") ?? 1e-5),
            [a + "attention.head_count"] = nHead,
            [a + "attention.kv_head_count"] = c.GetInt("num_key_value_heads") ?? nHead,
            [a + "attention.head_dim"] = headDim,
            [a + "rope.dim"] = RopeDim(c, headDim),
            [a + "rope.base"] = RopeBase(c),
            [a + "rope.style"] = "neox",
            [a + "mlp.activation"] = Activation(c),
        };
        if (c.GetInt("max_position_embeddings") is { } ctx) md[a + "context_length"] = ctx;
        if (type is "olmoe" or "qwen3_moe")
        {
            if (c.GetDouble("clip_qkv") is { } clip && clip > 0) throw new NotSupportedException("clip_qkv (qkv clamping) is not supported.");
            md[a + "moe.expert_count"] = c.Int("num_experts");
            md[a + "moe.expert_used_count"] = c.Int("num_experts_per_tok");
            md[a + "moe.ffn_hidden_size"] = c.GetInt("moe_intermediate_size") ?? c.Int("intermediate_size");
            md[a + "moe.normalize_weights"] = c.GetBool("norm_topk_prob") ?? false;
        }
        if (c.Get("rope_scaling") is { ValueKind: JsonValueKind.Object } && LongRope(c) is { } lr)
        {
            md[a + "rope.attention_scale"] = lr.AttentionScale;
            md[a + "rope.original_context_length"] = c.GetInt("original_max_position_embeddings") ?? c.GetInt("max_position_embeddings") ?? 4096;
        }
        else if (ComputeScaling(c) is { } sc && sc.AttentionScale != 1f) md[a + "rope.attention_scale"] = sc.AttentionScale;

        if (IsGemma(type))
        {
            md[a + "embedding_scale"] = MathF.Sqrt(hidden);
            if (c.GetDouble("query_pre_attn_scalar") is { } qpas) md[a + "attention.scale"] = (float)(1.0 / Math.Sqrt(qpas));
            if (c.GetDouble("attn_logit_softcapping") is { } asc && asc > 0) md[a + "attention.logit_softcap"] = (float)asc;
            if (c.GetDouble("final_logit_softcapping") is { } fsc && fsc > 0) md[a + "final_logit_softcap"] = (float)fsc;
            if (c.GetInt("sliding_window") is { } window && window > 0)
            {
                md[a + "attention.sliding_window"] = window;
                int[] sliding;
                if (c.GetStringArray("layer_types") is { } lt) sliding = lt.Select(t => t == "sliding_attention" ? 1 : 0).ToArray();
                else
                {
                    // gemma2: even layers slide; gemma3: every sliding_window_pattern-th layer is global.
                    int pattern = type == "gemma2" ? 2 : c.GetInt("sliding_window_pattern") ?? 6;
                    sliding = Enumerable.Range(0, nLayer).Select(il => il % pattern < pattern - 1 ? 1 : 0).ToArray();
                }
                md[a + "attention.sliding_layers"] = sliding;
                if (c.GetDouble("rope_local_base_freq") is { } local) md[a + "rope.local_base"] = (float)local;
                else if (type == "gemma3_text") md[a + "rope.local_base"] = 10000f;
            }
        }
        else if (type is not null && !type.StartsWith("qwen", StringComparison.Ordinal) && c.GetBool("use_sliding_window") != false
                 && c.GetInt("sliding_window") is { } swa && swa > 0)
            md[a + "attention.sliding_window"] = swa;   // every layer (Mistral v0.1, Phi-3-mini-4k); null / absent = full attention
        return md;
    }

    // ---- weights ------------------------------------------------------------------------------
    [GeneratedRegex(@"^model\.layers\.(\d+)\.(.+)$")]
    private static partial Regex LayerName();
    [GeneratedRegex(@"^mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")]
    private static partial Regex ExpertName();

    static readonly Dictionary<string, string> s_layer = new(StringComparer.Ordinal)
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

    // Norm tensors by HF name; Gemma's four-norm layout is renamed to the canonical roles.
    static readonly Dictionary<string, string> s_norms = new(StringComparer.Ordinal)
    {
        ["input_layernorm.weight"] = "input_norm.weight",
        ["self_attn.q_norm.weight"] = "attention.q_norm.weight",
        ["self_attn.k_norm.weight"] = "attention.k_norm.weight",
    };

    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HfImportOptions options)
    {
        bool extended = IsGemma4OrMuse(c.ModelType);
        bool gemma = IsGemma(c.ModelType) || extended;
        bool plusOne = IsGemma(c.ModelType) || c.ModelType == "muse_glimmer";   // Gemma 2/3 and Muse layer norms store w - 1 (Gemma 4's store w)
        // Gemma: post_attention_layernorm is the attention output norm, pre_feedforward_layernorm precedes the MLP.
        string postAttn = gemma ? "attention.out_norm.weight" : "post_attention_norm.weight";
        var result = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
        var unmapped = new List<string>();
        var experts = new SortedDictionary<(int Layer, string Kind), SortedDictionary<int, WeightInfo>>();
        void Add(WeightInfo w) { if (!result.TryAdd(w.Name, w)) throw new InvalidDataException($"Two HF tensors map to '{w.Name}'."); }
        WeightInfo Norm(WeightInfo w, string name) => plusOne ? HfTransforms.NormPlusOne(w, name) : HfTransforms.Plain(w, name);

        foreach (var (rawName, t) in tensors.OrderBy(p => p.Key, StringComparer.Ordinal))
        {
            string name = rawName;
            if (name.EndsWith("rotary_emb.inv_freq", StringComparison.Ordinal)) continue;
            var w = t.ToWeightInfo();
            if (extended)
            {
                if (IsNonTextTensor(name)) continue;
                name = name.Replace("model.language_model.", "model.");
            }
            switch (name)
            {
                case "model.embed_tokens.weight": Add(HfTransforms.Plain(w, "token_embedding.weight")); continue;
                case "model.embed_tokens_per_layer.weight" when extended: Add(HfTransforms.Plain(w, "per_layer.embedding.weight")); continue;
                case "model.per_layer_model_projection.weight" when extended: Add(HfTransforms.Plain(w, "per_layer.model_projection.weight")); continue;
                case "model.per_layer_projection_norm.weight" when extended: Add(HfTransforms.Plain(w, "per_layer.projection_norm.weight")); continue;
                case "model.norm.weight": Add(c.ModelType == "muse_glimmer" ? HfTransforms.Plain(w, "output_norm.weight") : Norm(w, "output_norm.weight")); continue;
                case "lm_head.weight":
                    if (!(c.GetBool("tie_word_embeddings") ?? false)) Add(HfTransforms.Plain(w, "lm_head.weight"));
                    continue;
            }
            var m = LayerName().Match(name);
            if (!m.Success) { unmapped.Add(name); continue; }
            string p = $"layers.{m.Groups[1].Value}.", suffix = m.Groups[2].Value;
            if (ExpertName().Match(suffix) is { Success: true } em)
            {
                var key = (int.Parse(m.Groups[1].Value), em.Groups[2].Value);
                if (!experts.TryGetValue(key, out var list)) experts[key] = list = [];
                list[int.Parse(em.Groups[1].Value)] = w;
            }
            else if (suffix == "mlp.gate.weight" && c.Has("num_experts")) Add(HfTransforms.Plain(w, p + "mlp.router.weight"));
            else if (s_layer.TryGetValue(suffix, out var cn)) Add(HfTransforms.Plain(w, p + cn));
            else if (s_norms.TryGetValue(suffix, out var nn)) Add(Norm(w, p + nn));
            else if (suffix == "post_attention_layernorm.weight") Add(Norm(w, p + postAttn));
            else if (gemma && suffix == "pre_feedforward_layernorm.weight") Add(Norm(w, p + "post_attention_norm.weight"));
            else if (suffix == "self_attn.qkv_proj.weight")
            {
                int nHead = c.Int("num_attention_heads"), nKv = c.GetInt("num_key_value_heads") ?? nHead, hd = HeadDim(c);
                Add(HfTransforms.Plain(w.RowSlice("", 0, nHead * hd), p + "attention.q.weight"));
                Add(HfTransforms.Plain(w.RowSlice("", nHead * hd, nKv * hd), p + "attention.k.weight"));
                Add(HfTransforms.Plain(w.RowSlice("", (nHead + nKv) * hd, nKv * hd), p + "attention.v.weight"));
            }
            else if (suffix == "mlp.gate_up_proj.weight")
            {
                int ff = c.Int("intermediate_size");
                Add(HfTransforms.Plain(w.RowSlice("", 0, ff), p + "mlp.gate.weight"));
                Add(HfTransforms.Plain(w.RowSlice("", ff, ff), p + "mlp.up.weight"));
            }
            else if (gemma && suffix == "post_feedforward_layernorm.weight") Add(Norm(w, p + "mlp.out_norm.weight"));
            else if (extended && s_extended.TryGetValue(suffix, out var xn)) Add(xn.EndsWith("norm.weight") ? Norm(w, p + xn) : HfTransforms.Plain(w, p + xn));
            else unmapped.Add(name);
        }
        foreach (var ((layer, kind), list) in experts)
        {
            if (list.Count != c.Int("num_experts") || list.Keys.Max() != list.Count - 1) throw new InvalidDataException($"layer {layer}: expected {c.Int("num_experts")} {kind} experts, found {list.Count}.");
            Add(WeightInfo.Stack($"layers.{layer}.mlp.experts.{kind}.weight", [.. list.Values]));
        }
        if (unmapped.Count > 0)
            throw new NotSupportedException($"{c.ModelType}: {unmapped.Count} tensor(s) have no canonical mapping: {string.Join(", ", unmapped.Take(10))}{(unmapped.Count > 10 ? ", ..." : "")}");

        if (extended)
        {
            if (ExtendedRopeFactors(c) is { } ef)
                Add(WeightInfo.Derived("rope.freq_factors", DataType.F32, new Shape(ef.Length), () => HfTransforms.EncodeF32(ef)));
            if (c.ModelType == "muse_glimmer")
                foreach (var (nm, v) in MuseQkNorms(c)) Add(nm);
        }
        else if (LongRope(c) is { } lr)
        {
            WeightInfo F32(string wn, float[] f) => WeightInfo.Derived(wn, DataType.F32, new Shape(f.Length), () => HfTransforms.EncodeF32(f));
            Add(F32("rope.freq_factors", lr.Short));
            Add(F32("rope.freq_factors_long", lr.Long));
        }
        else if (ComputeScaling(c) is { } sc)
        {
            var f = sc.Factors;
            Add(WeightInfo.Derived("rope.freq_factors", DataType.F32, new Shape(f.Length), () =>
            {
                var b = new byte[f.Length * 4];
                Buffer.BlockCopy(f, 0, b, 0, b.Length);
                return b;
            }));
        }
        return result;
    }
}
