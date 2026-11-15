using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// Xing4.0 (HF <c>xing4_0</c>, remote code): DeepSeek-V3's multi-head latent attention, dense first blocks and sigmoid-routed MoE layers with a
/// shared expert, on manifold-constrained hyper-connections (<c>hc_mult</c> streams). Its modules differ from DeepSeek-V4's: no eps on the
/// collapse weights, the mixing logits clamped to <c>mhc_h_res_clamp_min</c>/<c>max</c> before the row softmax (no eps after it), stream k
/// mixing the old streams with row k of the matrix, and the streams averaged before the output norm. The router picks the top experts
/// without groups.
/// </summary>
internal sealed partial class LatentDecoderHfMapping
{
    private static bool IsXing40(HfConfig c) => c.ModelType == "xing4_0";

    /// <exception cref="NotSupportedException">The mixing clamp is not symmetric.</exception>
    private static void AddXing40HyperParameters(Dictionary<string, object> md, string a, HfConfig c)
    {
        double clampMax = c.GetDouble("mhc_h_res_clamp_max") ?? 30;
        double clampMin = c.GetDouble("mhc_h_res_clamp_min") ?? -30;
        if (clampMin != -clampMax)
        {
            throw new NotSupportedException($"xing4_0: an asymmetric mixing clamp [{clampMin}, {clampMax}] is not supported.");
        }

        md[a + "hyper_connections.streams"] = c.GetInt("hc_mult") ?? 4;
        md[a + "hyper_connections.eps"] = (float)(c.GetDouble("hc_eps") ?? 1e-6);
        md[a + "hyper_connections.sinkhorn_iterations"] = c.GetInt("hc_sinkhorn_iters") ?? 20;
        md[a + "hyper_connections.pre_eps"] = 0f;
        md[a + "hyper_connections.mixing_eps"] = 0f;
        md[a + "hyper_connections.mixing_clamp"] = (float)clampMax;
        md[a + "hyper_connections.mixing_transposed"] = true;

        // The router takes the top experts of all (n_group / topk_group are not used).
        md[a + "moe.group_count"] = 1;
        md[a + "moe.group_used_count"] = 1;
    }

    /// <summary>The hyper-connection modules <c>attn_hc.hc_*</c> / <c>ffn_hc.hc_*</c>; false for other tensors.</summary>
    private static bool TryMapXing40Layer(string prefix, string suffix, WeightInfo w, HfWeightCollector collector)
        => suffix.StartsWith("attn_hc.hc_", StringComparison.Ordinal) || suffix.StartsWith("ffn_hc.hc_", StringComparison.Ordinal)
            ? TryMapGlm5NextLayer(prefix, suffix.Replace(".hc_", ".", StringComparison.Ordinal), w, collector)
            : false;
}
