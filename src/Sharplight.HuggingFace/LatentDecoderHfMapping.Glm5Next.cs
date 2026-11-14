using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// GLM-5.3-Flash (HF <c>glm5_next</c> / <c>glm5_next_text</c>, the language model): Kimi delta-attention layers with the bounded forget gate
/// (<c>linear_lower_bound</c>) and DSA layers whose multi-head latent attention has no rotary dimensions, a dense first block and MoE layers,
/// every MLP with the clamped SwiGLU (<c>swiglu_limit</c>), and DeepSeek-V4-style hyper-connections (<c>hc_mult</c> streams, Sinkhorn-normalized
/// mixing) whose streams are averaged before the output norm. Its k-pool indexer picks <c>index_topk</c> keys: whole pools of <c>index_kpool</c>
/// positions scored by their gated pooled keys, plus the query's incomplete pool (<c>index_kpool_always_select_tail</c>). The vision tower is
/// not imported.
/// </summary>
internal sealed partial class LatentDecoderHfMapping
{
    private static bool IsGlm5Next(HfConfig c) => c.ModelType is "glm5_next" or "glm5_next_text";

    private static void AddGlm5NextHyperParameters(Dictionary<string, object> md, string a, HfConfig c)
    {
        int layers = c.Int("num_hidden_layers");
        md[a + "hyper_connections.streams"] = c.Int("hc_mult");
        md[a + "hyper_connections.eps"] = (float)(c.GetDouble("hc_eps") ?? 1e-6);
        md[a + "hyper_connections.sinkhorn_iterations"] = c.GetInt("hc_sinkhorn_iters") ?? 20;
        if (c.GetDouble("swiglu_limit") is { } limit and > 0)
        {
            md[a + "mlp.swiglu_clamp"] = Enumerable.Repeat((float)limit, layers).ToArray();
            md[a + "mlp.swiglu_clamp_input"] = true;   // min(gate, limit) before the SiLU, up clamped to ±limit
        }

        md[a + "attention.indexer.token_budget"] = c.GetInt("index_topk") ?? 2048;
        md[a + "attention.indexer.pool_size"] = c.GetInt("index_kpool") ?? 16;
        md[a + "attention.indexer.pool_tail"] = c.GetBool("index_kpool_always_select_tail") ?? true;
    }

    /// <summary>GLM-5.3-Flash tensors that differ from DeepSeek's (hyper-connection modules); false for other tensors.</summary>
    private static bool TryMapGlm5NextLayer(string prefix, string suffix, WeightInfo w, HfWeightCollector collector)
    {
        string? canonical = suffix switch
        {
            "hc_attn_fn" or "attn_hc.fn" => "attention_hc.fn.weight",
            "hc_attn_base" or "attn_hc.base" => "attention_hc.base",
            "hc_attn_scale" or "attn_hc.scale" => "attention_hc.scale",
            "hc_ffn_fn" or "ffn_hc.fn" => "mlp_hc.fn.weight",
            "hc_ffn_base" or "ffn_hc.base" => "mlp_hc.base",
            "hc_ffn_scale" or "ffn_hc.scale" => "mlp_hc.scale",
            _ => null,
        };
        if (canonical is null)
        {
            return false;
        }

        collector.Add(HfTransforms.ToF32(w, prefix + canonical));
        return true;
    }
}
