using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face decoder-only transformers (<c>llama</c>, <c>mistral</c>, <c>qwen2</c>, <c>qwen3</c>, <c>gemma2</c>,
/// <c>gemma3_text</c>) → canonical <c>decoder</c> (vocabulary documented on <c>DecoderAdapter</c>). Weights are
/// mapped by presence (biases, q/k norms, Gemma's extra norms), Gemma's <c>1 + w</c> norms are folded, rope scaling
/// (llama3 / yarn / linear) becomes the canonical <c>rope.freq_factors</c> weight.
/// </summary>
public sealed partial class DecoderHfMapping : IHfArchitectureMapping
{
    public string Architecture => "decoder";
    public IReadOnlyList<string> ModelTypes { get; } = ["llama", "mistral", "qwen2", "qwen3", "gemma2", "gemma3_text"];

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

    static RopeScaling.Result? ComputeScaling(HfConfig c)
    {
        if (ScalingOf(c) is not { } s) return null;
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
        int nLayer = c.Int("num_hidden_layers");
        int nHead = c.Int("num_attention_heads");
        int hidden = c.Int("hidden_size");
        int headDim = HeadDim(c);
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = nLayer,
            [a + "hidden_size"] = hidden,
            [a + "ffn_hidden_size"] = c.Int("intermediate_size"),
            [a + "norm_eps"] = (float)c.Double("rms_norm_eps"),
            [a + "attention.head_count"] = nHead,
            [a + "attention.kv_head_count"] = c.GetInt("num_key_value_heads") ?? nHead,
            [a + "attention.head_dim"] = headDim,
            [a + "rope.dim"] = RopeDim(c, headDim),
            [a + "rope.base"] = RopeBase(c),
            [a + "rope.style"] = "neox",
            [a + "mlp.activation"] = Activation(c),
        };
        if (c.GetInt("max_position_embeddings") is { } ctx) md[a + "context_length"] = ctx;
        if (ComputeScaling(c) is { } sc && sc.AttentionScale != 1f) md[a + "rope.attention_scale"] = sc.AttentionScale;

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
        return md;
    }

    // ---- weights ------------------------------------------------------------------------------
    [GeneratedRegex(@"^model\.layers\.(\d+)\.(.+)$")]
    private static partial Regex LayerName();

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
        bool gemma = IsGemma(c.ModelType);
        // Gemma: post_attention_layernorm is the attention output norm, pre_feedforward_layernorm precedes the MLP.
        string postAttn = gemma ? "attention.out_norm.weight" : "post_attention_norm.weight";
        var result = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
        var unmapped = new List<string>();
        void Add(WeightInfo w) { if (!result.TryAdd(w.Name, w)) throw new InvalidDataException($"Two HF tensors map to '{w.Name}'."); }
        WeightInfo Norm(WeightInfo w, string name) => gemma ? HfTransforms.NormPlusOne(w, name) : HfTransforms.Plain(w, name);

        foreach (var (name, t) in tensors.OrderBy(p => p.Key, StringComparer.Ordinal))
        {
            if (name.EndsWith("rotary_emb.inv_freq", StringComparison.Ordinal)) continue;
            var w = t.ToWeightInfo();
            switch (name)
            {
                case "model.embed_tokens.weight": Add(HfTransforms.Plain(w, "token_embedding.weight")); continue;
                case "model.norm.weight": Add(Norm(w, "output_norm.weight")); continue;
                case "lm_head.weight":
                    if (!(c.GetBool("tie_word_embeddings") ?? false)) Add(HfTransforms.Plain(w, "lm_head.weight"));
                    continue;
            }
            var m = LayerName().Match(name);
            if (!m.Success) { unmapped.Add(name); continue; }
            string p = $"layers.{m.Groups[1].Value}.", suffix = m.Groups[2].Value;
            if (s_layer.TryGetValue(suffix, out var cn)) Add(HfTransforms.Plain(w, p + cn));
            else if (s_norms.TryGetValue(suffix, out var nn)) Add(Norm(w, p + nn));
            else if (suffix == "post_attention_layernorm.weight") Add(Norm(w, p + postAttn));
            else if (gemma && suffix == "pre_feedforward_layernorm.weight") Add(Norm(w, p + "post_attention_norm.weight"));
            else if (gemma && suffix == "post_feedforward_layernorm.weight") Add(Norm(w, p + "mlp.out_norm.weight"));
            else unmapped.Add(name);
        }
        if (unmapped.Count > 0)
            throw new NotSupportedException($"{c.ModelType}: {unmapped.Count} tensor(s) have no canonical mapping: {string.Join(", ", unmapped.Take(10))}{(unmapped.Count > 10 ? ", ..." : "")}");

        if (ComputeScaling(c) is { } sc)
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
