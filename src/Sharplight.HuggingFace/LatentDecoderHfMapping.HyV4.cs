using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hunyuan v4 (HF <c>hy_v4</c>): MLA with a sigmoid output gate and per-head attention sinks, a DSA indexer (only its dense regime:
/// the context is capped at <c>index_topk</c>, where it keeps every visible token, so its weights are not needed), dense first layers and
/// MoE layers with clamped SwiGLU experts, and hyper-connections without a mixing matrix (<c>hc_mult</c> streams, expansion weights
/// <c>hc_magnitude</c>·sigmoid + eps). Rotary dimensions use rotate-half pairs. A config with <c>hc_sinkhorn_iters</c> has DeepSeek-V4's modules
/// instead (fn / base with N² more rows for the Sinkhorn-normalized stream mixing matrix, a third scale, expansion weights without + eps).
/// </summary>
internal sealed partial class LatentDecoderHfMapping
{
    private static bool IsHyV4(HfConfig c) => c.ModelType == "hy_v4";

    private static void AddHyV4HyperParameters(Dictionary<string, object> md, string a, HfConfig c)
    {
        int layers = c.Int("num_hidden_layers");
        float eps = (float)(c.GetDouble("hc_eps") ?? 1e-6);
        md[a + "hyper_connections.streams"] = c.Int("hc_mult");
        md[a + "hyper_connections.eps"] = eps;
        md[a + "hyper_connections.magnitude"] = (float)(c.GetDouble("hc_magnitude") ?? 2.0);
        int sinkhorn = c.GetInt("hc_sinkhorn_iters") ?? 0;
        md[a + "hyper_connections.post_offset"] = sinkhorn > 0 ? 0f : eps;
        md[a + "hyper_connections.sinkhorn_iterations"] = sinkhorn;
        md[a + "attention.indexer.token_budget"] = c.GetInt("index_topk") ?? 2048;
        if (c.GetDouble("swiglu_limit") is { } limit and > 0)
        {
            md[a + "moe.swiglu_clamp"] = Enumerable.Repeat((float)limit, layers).ToArray();
            md[a + "moe.swiglu_clamp_input"] = true;   // min(gate, limit) before the SiLU, up clamped to ±limit
        }
    }

    /// <summary>Hunyuan v4 tensor names that differ from DeepSeek's, HF suffix to canonical suffix (null: not needed).</summary>
    private static readonly Dictionary<string, string?> s_hyV4Layer = new(StringComparer.Ordinal)
    {
        ["mlp.linear_gate.weight"] = "mlp.gate.weight",
        ["mlp.shared_experts.linear_gate.weight"] = "mlp.shared.gate.weight",
        ["self_attn.linear_gate.weight"] = "attention.gate.weight",
        ["hc_pre.hc_attn_layer.hc_fn"] = "attention_hc.fn.weight",
        ["hc_pre.hc_attn_layer.hc_base"] = "attention_hc.base",
        ["hc_pre.hc_attn_layer.hc_scale"] = "attention_hc.scale",
        ["hc_pre.hc_mlp_layer.hc_fn"] = "mlp_hc.fn.weight",
        ["hc_pre.hc_mlp_layer.hc_base"] = "mlp_hc.base",
        ["hc_pre.hc_mlp_layer.hc_scale"] = "mlp_hc.scale",
    };

    private static bool TryMapHyV4Layer(string prefix, string suffix, WeightInfo w, HfWeightCollector collector)
    {
        if (suffix.StartsWith("self_attn.indexer.", StringComparison.Ordinal))
        {
            return true;   // dense regime only
        }

        if (suffix == "self_attn.learnable_sink_param")
        {
            collector.Add(HfTransforms.ToF32(w, prefix + "attention.sinks"));
            return true;
        }

        if (!s_hyV4Layer.TryGetValue(suffix, out var renamed))
        {
            return false;
        }

        collector.Add(renamed!.Contains("_hc.", StringComparison.Ordinal) ? HfTransforms.ToF32(w, prefix + renamed) : HfTransforms.Plain(w, prefix + renamed));
        return true;
    }

    private static bool TryMapHyV4Global(string name, WeightInfo w, HfWeightCollector collector)
    {
        string? canonical = name switch
        {
            "model.hc_pre.hc_head.hc_head_fn" => "output_hc.fn.weight",
            "model.hc_pre.hc_head.hc_head_base" => "output_hc.base",
            "model.hc_pre.hc_head.hc_head_scale" => "output_hc.scale",
            _ => null,
        };
        if (canonical is null)
        {
            return false;
        }

        collector.Add(HfTransforms.ToF32(w, canonical));
        return true;
    }
}
