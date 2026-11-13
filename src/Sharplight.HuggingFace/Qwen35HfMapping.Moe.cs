using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// Mixture-of-experts layers (<c>qwen3_5_moe</c>, <c>qwen3_next</c>) and the Qwen3-Next fused linear-attention projections.
/// </summary>
internal sealed partial class Qwen35HfMapping
{
    private const string Qwen3NextModelType = "qwen3_next";

    [GeneratedRegex(@"^mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")]
    private static partial Regex ExpertName();

    /// <summary>MoE tensors that are only renamed, HF suffix to canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_moePlain = new(StringComparer.Ordinal)
    {
        ["mlp.gate.weight"] = "mlp.router.weight",
        ["mlp.experts.gate_up_proj"] = "mlp.experts.gate_up.weight",
        ["mlp.experts.down_proj"] = "mlp.experts.down.weight",
        ["mlp.shared_expert.gate_proj.weight"] = "mlp.shared.gate.weight",
        ["mlp.shared_expert.up_proj.weight"] = "mlp.shared.up.weight",
        ["mlp.shared_expert.down_proj.weight"] = "mlp.shared.down.weight",
        ["mlp.shared_expert_gate.weight"] = "mlp.shared.input_gate.weight",
    };

    /// <summary>Whether the config describes Qwen3-Next (fused qkvz / ba projections, grouped value heads, plain RoPE).</summary>
    private static bool IsQwen3Next(HfConfig c) => c.ModelType == Qwen3NextModelType;

    /// <summary>The <c>moe.*</c> hyper-parameters (none for dense checkpoints).</summary>
    /// <exception cref="NotSupportedException">Some layers are dense (mlp_only_layers / decoder_sparse_step).</exception>
    private static void AddMoeHyperParameters(HfConfig c, Dictionary<string, object> md, string a)
    {
        if (!c.Has("num_experts"))
        {
            return;
        }

        bool denseLayers = c.Get("mlp_only_layers") is { ValueKind: System.Text.Json.JsonValueKind.Array } dense && dense.GetArrayLength() > 0;
        if (denseLayers || (c.GetInt("decoder_sparse_step") ?? 1) != 1)
        {
            throw new NotSupportedException("qwen3_5: checkpoints that mix dense and MoE layers (mlp_only_layers, decoder_sparse_step) are not supported.");
        }

        md[a + "moe.expert_count"] = c.Int("num_experts");
        md[a + "moe.expert_used_count"] = c.Int("num_experts_per_tok");
        md[a + "moe.normalize_weights"] = c.GetBool("norm_topk_prob") ?? true;
    }

    /// <summary>Maps a MoE tensor of a layer (per-expert tensors are stacked later); false when the tensor is not a MoE tensor.</summary>
    private static bool TryMapMoeTensor(string prefix, string suffix, WeightInfo w, LayerTargets targets)
    {
        if (s_moePlain.TryGetValue(suffix, out var plainName))
        {
            targets.Collector.Add(HfTransforms.Plain(w, prefix + plainName));
            return true;
        }

        var m = ExpertName().Match(suffix);
        if (!m.Success)
        {
            return false;
        }

        targets.Experts.Add($"{prefix}mlp.experts.{m.Groups[2].Value}.weight", int.Parse(m.Groups[1].Value), w);
        return true;
    }

    /// <summary>
    /// Qwen3-Next fused projections. <c>in_proj_qkvz</c> rows are per key head [q (dk), k (dk), v (r·dv), z (r·dv)] with r value
    /// heads per key head; the canonical qkv is [all q | all k | all v] and z separate, value heads in grouped order.
    /// <c>in_proj_ba</c> rows are per key head [b (r), a (r)].
    /// </summary>
    private static IReadOnlyList<WeightInfo>? MapFusedLinearAttention(string prefix, string suffix, WeightInfo w, LinearAttentionShape x)
    {
        int r = x.ValueHeadsPerKeyHead;
        switch (suffix)
        {
            case "linear_attn.in_proj_qkvz.weight":
            {
                long dk = x.KeyHeadDim;
                long dvGroup = (long)r * x.ValueHeadDim;
                long stride = 2 * dk + 2 * dvGroup;
                var heads = Enumerable.Range(0, x.KeyHeads).Select(h => h * stride).ToArray();
                var qkv = heads.Select(s => (s, dk))
                    .Concat(heads.Select(s => (s + dk, dk)))
                    .Concat(heads.Select(s => (s + 2 * dk, dvGroup)));
                var z = heads.Select(s => (s + 2 * dk + dvGroup, dvGroup));
                return [w.GatherRows(prefix + "linear_attention.qkv.weight", qkv), w.GatherRows(prefix + "linear_attention.z.weight", z)];
            }
            case "linear_attn.in_proj_ba.weight":
            {
                var heads = Enumerable.Range(0, x.KeyHeads).Select(h => (long)h * 2 * r).ToArray();
                return
                [
                    w.GatherRows(prefix + "linear_attention.beta.weight", heads.Select(s => (s, (long)r))),
                    w.GatherRows(prefix + "linear_attention.alpha.weight", heads.Select(s => (s + r, (long)r))),
                ];
            }
            default:
                return null;
        }
    }
}
