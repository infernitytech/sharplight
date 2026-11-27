using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.Gguf;

/// <summary>
/// llama.cpp <c>raon1</c> (Raon-1 mini): Raon delta attention layers (<c>attention.head_count_kv</c> 0; Kimi-Linear's tensors with separate q / k / v
/// convolutions, the low-rank decay <c>ssm_f_a</c> / <c>ssm_f_b</c>, the bounded decay <c>kda.gate_lower_bound</c>, a full-rank output gate
/// <c>ssm_g</c> and the write gate <c>ssm_rda</c>), NoPE latent attention with a sigmoid output gate (<c>attn_gate</c>; the rope dimensions are not
/// rotated), a dense SiTU-GLU first block (<c>leading_dense_block_count</c>) and sigmoid-routed SiTU MoE with shared experts, on Raon_mHC
/// hyper-connections (<c>hyper_connection.{count,epsilon,magnitude}</c>, <c>hc_{attn,ffn}_{fn,base,scale}</c> with the Helmert-chart mixing) whose
/// streams are averaged before the output norm. KRAFTON's files also carry reasoning-effort steering vectors (<c>ffn_effort_steer</c>,
/// <c>reasoning_effort.{names,alpha,default}</c>), which llama.cpp does not read; Sharplight applies the default effort's factor.
/// </summary>
internal sealed partial class LatentDecoderGgufMapping
{
    private static bool IsRaon1(GgufMetadataView g) => g.GgufArchitecture == "raon1";

    private static void AddRaon1HyperParameters(Dictionary<string, object> md, string a, GgufMetadataView g)
    {
        md[a + "rope.style"] = "none";
        md[a + "attention.latent_norm_eps"] = g.Float("attention.latent_norm_rms_epsilon", 1e-6f);
        md[a + "mlp.activation"] = "situ";
        md[a + "mlp.situ_beta"] = g.Float("activation.situ_beta");
        md[a + "mlp.situ_linear_beta"] = g.Float("activation.situ_linear_beta");
        if (g.Has("kda.gate_lower_bound"))
        {
            md[a + "linear_attention.decay_bound"] = g.Float("kda.gate_lower_bound");
        }

        md[a + "hyper_connections.streams"] = g.Int("hyper_connection.count");
        md[a + "hyper_connections.norm_eps"] = g.Float("hyper_connection.epsilon", 1e-5f);
        md[a + "hyper_connections.magnitude"] = g.Float("hyper_connection.magnitude", 2f);
        md[a + "hyper_connections.eps"] = 0f;
        md[a + "hyper_connections.pre_eps"] = 0f;
        md[a + "hyper_connections.chart_mixing"] = true;
        md[a + "hyper_connections.mixing_transposed"] = true;
        if (DefaultEffortScale(g) is { } scale and not 0f)
        {
            md[a + "mlp.steering_scale"] = scale;
        }
    }

    /// <summary>The factor of <c>reasoning_effort.default</c> among <c>reasoning_effort.names</c> / <c>alpha</c>; null without steering keys.</summary>
    private static float? DefaultEffortScale(GgufMetadataView g)
    {
        if (g.Raw("reasoning_effort.default") is not string effort || g.Raw("reasoning_effort.names") is not Array names
            || g.Raw("reasoning_effort.alpha") is not Array alpha)
        {
            return null;
        }

        for (int i = 0; i < names.Length && i < alpha.Length; i++)
        {
            if (names.GetValue(i) as string == effort)
            {
                return Convert.ToSingle(alpha.GetValue(i), System.Globalization.CultureInfo.InvariantCulture);
            }
        }

        return null;
    }

    /// <summary>The per-layer tensors Raon-1 adds, GGUF suffix to canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_raon1 = new(StringComparer.Ordinal)
    {
        ["attn_gate.weight"] = "attention.gate.weight",
        ["ffn_effort_steer.weight"] = "mlp.steering",
    };

    /// <summary>The RDA write gate (only on delta-attention layers).</summary>
    private const string RaonWriteGate = "ssm_rda.weight";

    /// <summary>The f32 form of the reasoning-effort steering vectors; null for other tensors.</summary>
    private static WeightInfo? Raon1Vector(WeightInfo w)
    {
        if (!w.Name.EndsWith(".mlp.steering", StringComparison.Ordinal) || w.Type == DataType.F32)
        {
            return null;
        }

        return WeightInfo.Derived(w.Name, DataType.F32, new Shape(w.Shape.Elements()),
            () => System.Runtime.InteropServices.MemoryMarshal.AsBytes(GgmlBlockDecoder.Decode(w.Type, w.ReadAll()).AsSpan()).ToArray());
    }
}
