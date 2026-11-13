using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>Gemma 4 and Muse Glimmer text models: the generic decoder plus per-layer head sizes, KV sharing, per-layer embeddings, NoPE layers, ...</summary>
public sealed partial class DecoderHfMapping
{
    static bool IsGemma4OrMuse(string? t) => t is "gemma4" or "gemma4_text" or "muse_glimmer";

    /// <summary>Vision / audio towers and their projectors live in other canonical sources.</summary>
    static bool IsNonTextTensor(string name)
        => name.StartsWith("model.vision_tower.", StringComparison.Ordinal) || name.StartsWith("model.audio_tower.", StringComparison.Ordinal)
        || name.StartsWith("model.embed_vision.", StringComparison.Ordinal) || name.StartsWith("model.embed_audio.", StringComparison.Ordinal)
        || name.StartsWith("model.vision_", StringComparison.Ordinal) || name.StartsWith("model.visual.", StringComparison.Ordinal);

    static readonly Dictionary<string, string> s_extended = new(StringComparer.Ordinal)
    {
        ["self_attn.gate_proj.weight"] = "attention.gate.weight",
        ["per_layer_input_gate.weight"] = "per_layer.gate.weight",
        ["per_layer_projection.weight"] = "per_layer.projection.weight",
        ["post_per_layer_input_norm.weight"] = "per_layer.norm.weight",
        ["layer_scalar"] = "output_scale",
    };

    static JsonElement? LayerTypeRope(HfConfig c, string layerType)
        => c.RopeParameters is { } rp && rp.TryGetProperty(layerType, out var e) && e.ValueKind == JsonValueKind.Object ? e : null;

    static double RopeNumber(JsonElement? e, string key, double def)
        => e is { } x && x.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetDouble() : def;

    static string[] LayerTypes(HfConfig c, int nLayer)
        => c.GetStringArray("layer_types") ?? Enumerable.Repeat("full_attention", nLayer).ToArray();

    /// <summary>Per-frequency divisors of the global layers' rope (Gemma 4 proportional rotary), or null.</summary>
    static float[]? ExtendedRopeFactors(HfConfig c)
    {
        if (c.ModelType is not ("gemma4" or "gemma4_text")) return null;
        var full = LayerTypeRope(c, "full_attention");
        if (full is not { } f || !f.TryGetProperty("rope_type", out var rt) || rt.GetString() != "proportional") return null;
        int dim = c.GetInt("global_head_dim") ?? c.Int("head_dim");
        int angles = (int)(RopeNumber(f, "partial_rotary_factor", 1.0) * dim / 2);
        return Sharplight.Models.RopeScaling.Proportional(dim, angles, (float)RopeNumber(f, "factor", 1.0)).Factors;
    }

    IReadOnlyDictionary<string, object> Gemma4Hyper(HfConfig c)
    {
        if (c.GetBool("enable_moe_block") == true) throw new NotSupportedException("Gemma 4 mixture-of-experts blocks are not supported yet.");
        string a = Architecture + ".";
        int nLayer = c.Int("num_hidden_layers"), nHead = c.Int("num_attention_heads"), hidden = c.Int("hidden_size");
        int headDim = c.Int("head_dim"), globalDim = c.GetInt("global_head_dim") ?? headDim;
        int kv = c.GetInt("num_key_value_heads") ?? nHead;
        int globalKv = c.GetBool("attention_k_eq_v") == true ? c.GetInt("num_global_key_value_heads") ?? kv : kv;
        var types = LayerTypes(c, nLayer);
        var full = LayerTypeRope(c, "full_attention");
        var sliding = LayerTypeRope(c, "sliding_attention");
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = nLayer,
            [a + "hidden_size"] = hidden,
            [a + "ffn_hidden_size"] = c.Int("intermediate_size"),
            [a + "norm_eps"] = (float)(c.GetDouble("rms_norm_eps") ?? 1e-6),
            [a + "attention.head_count"] = nHead,
            [a + "attention.kv_head_count"] = kv,
            [a + "attention.head_dim"] = headDim,
            [a + "attention.scale"] = 1f,
            [a + "attention.v_norm"] = true,
            [a + "attention.layer_head_dims"] = types.Select(t => t == "full_attention" ? globalDim : headDim).ToArray(),
            [a + "attention.layer_kv_head_counts"] = types.Select(t => t == "full_attention" ? globalKv : kv).ToArray(),
            [a + "attention.sliding_window"] = c.GetInt("sliding_window") ?? 0,
            [a + "attention.sliding_layers"] = types.Select(t => t == "sliding_attention" ? 1 : 0).ToArray(),
            [a + "rope.dim"] = globalDim,
            [a + "rope.local_dim"] = headDim,
            [a + "rope.base"] = (float)RopeNumber(full, "rope_theta", 1000000.0),
            [a + "rope.local_base"] = (float)RopeNumber(sliding, "rope_theta", 10000.0),
            [a + "rope.style"] = "neox",
            [a + "mlp.activation"] = Activation(c),
            [a + "embedding_scale"] = MathF.Sqrt(hidden),
        };
        if (c.GetInt("max_position_embeddings") is { } ctx) md[a + "context_length"] = ctx;
        if (c.GetDouble("final_logit_softcapping") is { } fsc && fsc > 0) md[a + "final_logit_softcap"] = (float)fsc;
        int shared = c.GetInt("num_kv_shared_layers") ?? 0;
        if (shared > 0)
        {
            int first = nLayer - shared;
            var source = new int[nLayer];
            for (int il = 0; il < nLayer; il++)
                source[il] = il < first ? il : Enumerable.Range(0, first).Last(j => types[j] == types[il]);
            md[a + "attention.kv_source_layers"] = source;
        }
        if (c.GetInt("hidden_size_per_layer_input") is { } ple and > 0) md[a + "per_layer.size"] = ple;
        return md;
    }

    IReadOnlyDictionary<string, object> MuseHyper(HfConfig c)
    {
        string a = Architecture + ".";
        int nLayer = c.Int("num_hidden_layers"), nHead = c.Int("num_attention_heads"), hidden = c.Int("hidden_size");
        int headDim = c.GetInt("head_dim") ?? hidden / nHead;
        var types = LayerTypes(c, nLayer);
        double theta = RopeNumber(c.RopeParameters, "rope_theta", 10000.0);
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = nLayer,
            [a + "hidden_size"] = hidden,
            [a + "ffn_hidden_size"] = c.Int("intermediate_size"),
            [a + "norm_eps"] = (float)(c.GetDouble("rms_norm_eps") ?? 1e-5),
            [a + "post_norm_eps"] = (float)(c.GetDouble("post_norm_eps") ?? 1e-5),
            [a + "attention.head_count"] = nHead,
            [a + "attention.kv_head_count"] = c.GetInt("num_key_value_heads") ?? nHead,
            [a + "attention.head_dim"] = headDim,
            [a + "attention.scale"] = (float)((c.GetDouble("qk_scale_factor") ?? 1.0) / Math.Sqrt(headDim)),   // q is scaled by qk_scale_factor after its norm
            [a + "attention.qk_norm_unweighted"] = true,
            [a + "attention.sliding_window"] = c.GetInt("sliding_window") ?? 0,
            [a + "attention.sliding_layers"] = types.Select(t => t == "sliding_attention" ? 1 : 0).ToArray(),
            [a + "rope.dim"] = headDim,
            [a + "rope.base"] = (float)theta,
            [a + "rope.style"] = "neox",
            [a + "mlp.activation"] = Activation(c),
            [a + "embedding_norm"] = true,
        };
        if (c.Get("layer_rope_theta") is { ValueKind: JsonValueKind.Array } lrt)
            md[a + "rope.layers"] = lrt.EnumerateArray().Select(e => e.GetDouble() != 0 ? 1 : 0).ToArray();
        if (c.GetInt("max_position_embeddings") is { } ctx) md[a + "context_length"] = ctx;
        if (c.GetDouble("output_multiplier") is { } om && om != 1.0) md[a + "logit_scale"] = (float)om;
        if (c.GetDouble("final_logit_softcapping") is { } fsc && fsc > 0) md[a + "final_logit_softcap"] = (float)fsc;
        return md;
    }
}
