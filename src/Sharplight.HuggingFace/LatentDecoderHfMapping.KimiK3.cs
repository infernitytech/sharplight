using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.HuggingFace;

/// <summary>
/// Kimi K3 (HF <c>kimi_k3</c>, the language model below <c>language_model.</c>; the vision tower is not imported): Kimi-Linear
/// layers with block attention residuals (<c>attn_res_block_size</c>; the per-layer <c>*_res_norm</c> / <c>*_res_proj</c> pairs become one score vector
/// each), the situ activation in every MLP (<c>activation_situ_beta</c> / <c>activation_situ_linear_beta</c>), latent MoE (<c>routed_expert_hidden_size</c>,
/// <c>routed_expert_norm</c> after the expert sum), the bounded delta-attention gate (<c>linear_attn_config.gate_lower_bound</c>) with a full-rank output
/// gate and one decay rate per key channel shared by the heads (<c>A_log</c> [head_dim]), and a sigmoid output gate on the latent attention.
/// </summary>
internal sealed partial class LatentDecoderHfMapping
{
    private static bool IsKimiK3(HfConfig c) => c.ModelType == "kimi_k3";

    private static void AddKimiK3HyperParameters(Dictionary<string, object> md, string a, HfConfig c)
    {
        if (c.GetInt("attn_res_block_size") is { } blockSize and > 0)
        {
            md[a + "attention_residuals.block_size"] = blockSize;
        }

        if (c.GetString("hidden_act") == "situ")
        {
            // The reference uses beta "or 1.0" and leaves the up projection alone without a linear beta.
            md[a + "mlp.activation"] = "situ";
            md[a + "mlp.situ_beta"] = (float)(c.GetDouble("activation_situ_beta") is { } beta and not 0 ? beta : 1.0);
            md[a + "mlp.situ_linear_beta"] = (float)(c.GetDouble("activation_situ_linear_beta") ?? 0.0);
        }

        // The query and latent norms of the latent attention keep the reference's default epsilon.
        md[a + "attention.latent_norm_eps"] = 1e-6f;
        if (HfMappingHelpers.OptionalNumber(LinearConfig(c), "gate_lower_bound") is { } bound)
        {
            md[a + "linear_attention.decay_bound"] = (float)bound;
        }
    }

    /// <summary>The text model's tensor name (without <c>language_model.</c>), or null for tensors that are not imported (vision tower, projector).</summary>
    private static string? KimiK3TextName(string name)
    {
        if (name.StartsWith("vision_tower.", StringComparison.Ordinal) || name.StartsWith("mm_projector.", StringComparison.Ordinal))
        {
            return null;
        }

        return name.StartsWith("language_model.", StringComparison.Ordinal) ? name["language_model.".Length..] : name;
    }

    /// <summary>The attention-residual score parts: HF prefix (<c>model.layers.{i}.self_attention_res</c>, ...) to its canonical score name.</summary>
    private static string? ResidualScoreName(string name, out string part)
    {
        part = "";
        foreach (string kind in (string[])["_norm.weight", "_proj.weight"])
        {
            if (!name.EndsWith(kind, StringComparison.Ordinal))
            {
                continue;
            }

            string stem = name[..^kind.Length];
            part = kind;
            if (stem == "model.output_attn_res")
            {
                return "output_res.score.weight";
            }

            var m = LayerName().Match(stem);
            if (m.Success && m.Groups[2].Value is "self_attention_res" or "mlp_res")
            {
                return $"layers.{m.Groups[1].Value}.{(m.Groups[2].Value == "mlp_res" ? "mlp_res" : "attention_res")}.score.weight";
            }
        }

        return null;
    }

    /// <summary>Collects one half of an attention-residual score (norm weight or 1-row projection); false for other tensors.</summary>
    private static bool TryCollectResidualScore(string name, WeightInfo w, Dictionary<string, (WeightInfo? Norm, WeightInfo? Projection)> parts)
    {
        if (ResidualScoreName(name, out string part) is not { } score)
        {
            return false;
        }

        parts.TryGetValue(score, out var pair);
        parts[score] = part == "_norm.weight" ? (w, pair.Projection) : (pair.Norm, w);
        return true;
    }

    /// <summary>Every score vector: the elementwise product of its norm weight and projection, f32 [H].</summary>
    /// <exception cref="InvalidDataException">A norm weight or projection is missing.</exception>
    private static void AddResidualScores(Dictionary<string, (WeightInfo? Norm, WeightInfo? Projection)> parts, HfWeightCollector collector)
    {
        foreach (var (name, (norm, projection)) in parts.OrderBy(p => p.Key, StringComparer.Ordinal))
        {
            if (norm is null || projection is null)
            {
                throw new InvalidDataException($"{name}: an attention residual needs both its norm weight and its projection.");
            }

            long hidden = norm.Shape.Elements();
            collector.Add(WeightInfo.Derived(name, DataType.F32, new Shape(hidden), () =>
            {
                var n = HfTransforms.DecodeF32(norm);
                var p = HfTransforms.DecodeF32(projection);
                return HfTransforms.EncodeF32([.. n.Select((v, i) => v * p[i])]);
            }));
        }
    }

    /// <summary>Latent-MoE tensors below <c>block_sparse_moe.</c> (after <see cref="NormalizeDeltaHybridSuffix"/>: <c>mlp.</c>); false for others.</summary>
    private static bool TryMapLatentMoe(string prefix, string suffix, WeightInfo w, HfWeightCollector collector)
    {
        string? canonical = suffix switch
        {
            "mlp.routed_expert_down_proj.weight" => "mlp.latent_down.weight",
            "mlp.routed_expert_up_proj.weight" => "mlp.latent_up.weight",
            "mlp.routed_expert_norm.weight" => "mlp.latent_out_norm.weight",
            _ => null,
        };
        if (canonical is null)
        {
            return false;
        }

        collector.Add(HfTransforms.Plain(w, prefix + canonical));
        return true;
    }

    /// <summary>K3's decay rate per key channel, shared by the heads: −exp(A_log) [D] repeated for every head, f32 [heads·D].</summary>
    private static WeightInfo ChannelDecayRates(HfConfig c, WeightInfo w, string name)
    {
        int heads = LinearHeads(c);
        long dims = w.Shape.Elements();
        return WeightInfo.Derived(name, DataType.F32, new Shape(heads * dims), () =>
        {
            var rates = HfTransforms.DecodeF32(w).Select(static v => -MathF.Exp(v)).ToArray();
            return HfTransforms.EncodeF32([.. Enumerable.Range(0, heads).SelectMany(_ => rates)]);
        });
    }
}
