using System.Runtime.InteropServices;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Gguf;

/// <summary>
/// Delta-attention hybrids on the latent decoder: <c>kimi-linear</c> (Kimi-Linear: low-rank decay / gate projections, latent attention without
/// rotary embedding) and <c>bailingmoe3</c> (Ling 3: full-rank projections, the bounded gate, latent attention with rope and a per-head output
/// gate). A layer whose <c>attention.head_count_kv</c> entry is 0 is a delta-attention layer; its three short convolutions
/// (<c>ssm_conv1d_{q,k,v}</c>) are joined into the canonical [3·heads·D, kernel] f32 weight.
/// </summary>
internal sealed partial class LatentDecoderGgufMapping
{
    private static bool IsKimiLinear(GgufMetadataView g) => g.GgufArchitecture == "kimi-linear";

    private static bool IsBailing3(GgufMetadataView g) => g.GgufArchitecture == "bailingmoe3";

    private static bool IsDeltaHybrid(GgufMetadataView g) => IsKimiLinear(g) || IsBailing3(g);

    /// <summary>Whether trunk layer <paramref name="layer"/> is a delta-attention layer (no key/value heads).</summary>
    private static bool IsDeltaLayer(GgufMetadataView g, int layer)
        => g.Raw("attention.head_count_kv") is Array values && layer < values.Length && Convert.ToInt32(values.GetValue(layer)) == 0;

    private static void AddDeltaHyperParameters(Dictionary<string, object> md, string a, GgufMetadataView g)
    {
        md[a + "linear_attention.head_count"] = g.Int("attention.head_count");
        md[a + "linear_attention.head_dim"] = g.Int("kda.head_dim");
        md[a + "linear_attention.conv_kernel"] = g.Int("ssm.conv_kernel");
        md[a + "linear_attention.l2_eps"] = 1e-6f;
        if (IsBailing3(g))
        {
            md[a + "linear_attention.decay_bound"] = g.Float("kda.gate_lower_bound");
        }
        else
        {
            md[a + "rope.style"] = "none";               // Kimi-Linear's latent attention has no rotary embedding
            md[a + "moe.normalize_weights"] = true;      // llama.cpp renormalizes Kimi-Linear's routing weights unconditionally
        }
    }

    /// <summary>Delta-attention tensors, GGUF suffix to canonical suffix (the decay and gate inputs differ by architecture).</summary>
    private static readonly Dictionary<string, string> s_delta = new(StringComparer.Ordinal)
    {
        ["attn_q.weight"] = "linear_attention.q.weight",
        ["attn_k.weight"] = "linear_attention.k.weight",
        ["attn_v.weight"] = "linear_attention.v.weight",
        ["attn_output.weight"] = "linear_attention.out.weight",
        ["ssm_beta.weight"] = "linear_attention.beta.weight",
        ["ssm_dt.bias"] = "linear_attention.dt_bias",
        ["ssm_a"] = "linear_attention.a",
        ["ssm_norm.weight"] = "linear_attention.norm.weight",
        ["ssm_conv1d_q.weight"] = ConvPart + "q",
        ["ssm_conv1d_k.weight"] = ConvPart + "k",
        ["ssm_conv1d_v.weight"] = ConvPart + "v",
    };

    /// <summary>Canonical names of the separate convolutions, joined by <see cref="JoinedConvolutions"/>.</summary>
    private const string ConvPart = "linear_attention.conv1d_part.";

    /// <summary>Canonical suffix of a delta-layer tensor, or null when it is not delta-attention specific.</summary>
    private static string? MapDeltaSuffix(GgufMetadataView g, string suffix)
    {
        bool lowRank = IsKimiLinear(g);
        return suffix switch
        {
            "ssm_f_a.weight" => lowRank ? "linear_attention.decay_down.weight" : "linear_attention.decay.weight",
            "ssm_f_b.weight" => "linear_attention.decay_up.weight",
            "ssm_g_a.weight" => lowRank ? "linear_attention.gate_down.weight" : "linear_attention.gate.weight",
            "ssm_g_b.weight" => "linear_attention.gate_up.weight",
            _ => s_delta.GetValueOrDefault(suffix),
        };
    }

    /// <summary>
    /// The canonical f32 form of a delta-layer vector: <c>a</c> is −exp(A_log) (Kimi-Linear stores it so; Ling 3 stores exp(A_log)), the decay
    /// bias flattened. Null for other tensors.
    /// </summary>
    private static WeightInfo? DeltaVector(WeightInfo w, GgufMetadataView g)
    {
        bool a = w.Name.EndsWith(".linear_attention.a", StringComparison.Ordinal);
        if (!a && !w.Name.EndsWith(".linear_attention.dt_bias", StringComparison.Ordinal))
        {
            return null;
        }

        float sign = a && IsBailing3(g) ? -1f : 1f;
        long count = w.Shape.Elements();
        return WeightInfo.Derived(w.Name, DataType.F32, new Shape(count), () =>
        {
            var values = GgmlBlockDecoder.Decode(w.Type, w.ReadAll());
            for (int i = 0; i < values.Length; i++)
            {
                values[i] *= sign;
            }

            return MemoryMarshal.AsBytes(values.AsSpan()).ToArray();
        });
    }

    /// <summary>The joined [3·heads·D, kernel] f32 convolution weight of every delta-attention layer.</summary>
    /// <exception cref="InvalidDataException">A layer lacks one of its q / k / v convolutions.</exception>
    private static IEnumerable<WeightInfo> JoinedConvolutions(IReadOnlyDictionary<string, WeightInfo> weights)
    {
        foreach (var (name, q) in weights.Where(p => p.Key.EndsWith(ConvPart + "q", StringComparison.Ordinal)).OrderBy(p => p.Key, StringComparer.Ordinal))
        {
            string prefix = name[..^(ConvPart.Length + 1)];
            if (!weights.TryGetValue(prefix + ConvPart + "k", out var k) || !weights.TryGetValue(prefix + ConvPart + "v", out var v))
            {
                throw new InvalidDataException($"{prefix}: delta attention needs q, k and v convolution weights.");
            }

            var parts = new[] { q, k, v };
            long kernel = q.Shape[-1].Fixed;   // [1, channels, 1, kernel]
            long channels = parts.Sum(p => p.Shape.Elements() / kernel);
            yield return WeightInfo.Derived(prefix + "linear_attention.conv1d.weight", DataType.F32, new Shape(channels, kernel), () =>
            {
                var joined = parts.SelectMany(p => GgmlBlockDecoder.Decode(p.Type, p.ReadAll())).ToArray();
                return MemoryMarshal.AsBytes(joined.AsSpan()).ToArray();
            });
        }
    }
}
