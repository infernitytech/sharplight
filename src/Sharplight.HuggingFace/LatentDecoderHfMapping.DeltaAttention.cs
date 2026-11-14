using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.HuggingFace;

/// <summary>
/// Delta-attention hybrids on the latent decoder: <c>kimi_linear</c> (Kimi-Linear: delta attention with low-rank decay / gate projections, latent
/// attention without rotary embedding, <c>block_sparse_moe</c> experts named w1 / w3 / w2) and <c>bailing_hybrid</c> (Ling 3: delta attention with the
/// bounded gate and full-rank projections, latent attention with rope and a per-head output gate, <c>attention.*</c> / <c>dense</c> names).
/// Layer kinds come from <c>layer_types</c> (or Kimi's 1-based <c>linear_attn_config.kda_layers</c>); the three short convolutions of a delta layer
/// are joined into one [3·heads·D, kernel] weight.
/// </summary>
internal sealed partial class LatentDecoderHfMapping
{
    private static bool IsKimiLinear(HfConfig c) => c.ModelType == "kimi_linear";

    private static bool IsBailingHybrid(HfConfig c) => c.ModelType == "bailing_hybrid";

    private static bool IsDeltaHybrid(HfConfig c) => IsKimiLinear(c) || IsBailingHybrid(c);

    private static JsonElement? LinearConfig(HfConfig c) => c.Get("linear_attn_config") is { ValueKind: JsonValueKind.Object } j ? j : null;

    private static int LinearHeads(HfConfig c)
        => (int)(HfMappingHelpers.OptionalNumber(LinearConfig(c), "num_heads") ?? c.GetInt("linear_num_heads") ?? c.Int("num_attention_heads"));

    private static int LinearHeadDim(HfConfig c)
        => (int)(HfMappingHelpers.OptionalNumber(LinearConfig(c), "head_dim") ?? c.GetInt("linear_head_dim") ?? c.Int("head_dim"));

    private static int LinearKernel(HfConfig c)
        => (int)(HfMappingHelpers.OptionalNumber(LinearConfig(c), "short_conv_kernel_size") ?? c.GetInt("linear_conv_kernel_dim", "short_conv_kernel_size") ?? 4);

    /// <summary>Whether trunk layer <paramref name="layer"/> is a delta-attention layer.</summary>
    private static bool IsDeltaLayer(HfConfig c, int layer)
    {
        if (c.GetStringArray("layer_types") is { } types && layer < types.Length)
        {
            return types[layer] == "linear_attention";
        }

        if (LinearConfig(c) is { } linear && linear.TryGetProperty("kda_layers", out var kda) && kda.ValueKind == JsonValueKind.Array)
        {
            return kda.EnumerateArray().Any(e => e.GetInt32() == layer + 1);   // 1-based
        }

        if (IsBailingHybrid(c))
        {
            return (layer + 1) % (c.GetInt("layer_group_size") ?? 6) != 0;
        }

        return layer % 4 != 3;
    }

    private static void AddDeltaHyperParameters(Dictionary<string, object> md, string a, HfConfig c)
    {
        md[a + "linear_attention.head_count"] = LinearHeads(c);
        md[a + "linear_attention.head_dim"] = LinearHeadDim(c);
        md[a + "linear_attention.conv_kernel"] = LinearKernel(c);
        md[a + "linear_attention.l2_eps"] = 1e-6f;
        if (IsBailingHybrid(c) && (c.GetBool("kda_safe_gate") ?? true))
        {
            md[a + "linear_attention.decay_bound"] = (float)(c.GetDouble("kda_lower_bound") ?? -5.0);
        }

        if (IsKimiLinear(c))
        {
            md[a + "rope.style"] = "none";   // the latent attention layers apply no rotary embedding
        }
    }

    [GeneratedRegex(@"^(block_sparse_moe|mlp)\.experts\.(\d+)\.w([123])\.weight$")]
    private static partial Regex KimiExpertName();

    /// <summary>Renames the suffixes of these checkpoints to the DeepSeek-style suffixes <see cref="TryMapLayer"/> knows.</summary>
    private static string NormalizeDeltaHybridSuffix(HfConfig c, string suffix)
    {
        if (IsKimiLinear(c))
        {
            if (KimiExpertName().Match(suffix) is { Success: true } e)
            {
                string kind = e.Groups[3].Value switch { "1" => "gate", "3" => "up", _ => "down" };
                return $"mlp.experts.{e.Groups[2].Value}.{kind}_proj.weight";
            }

            return suffix.StartsWith("block_sparse_moe.", StringComparison.Ordinal) ? "mlp." + suffix["block_sparse_moe.".Length..] : suffix;
        }

        if (suffix.StartsWith("attention.", StringComparison.Ordinal))
        {
            suffix = "self_attn." + suffix["attention.".Length..];
        }

        return suffix switch
        {
            "self_attn.dense.weight" => "self_attn.o_proj.weight",
            "mlp.gate.expert_bias" => "mlp.gate.e_score_correction_bias",
            _ => suffix,
        };
    }

    /// <summary>Delta-attention suffix (after <see cref="NormalizeDeltaHybridSuffix"/>) to canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_delta = new(StringComparer.Ordinal)
    {
        ["self_attn.q_proj.weight"] = "linear_attention.q.weight",
        ["self_attn.k_proj.weight"] = "linear_attention.k.weight",
        ["self_attn.v_proj.weight"] = "linear_attention.v.weight",
        ["self_attn.f_a_proj.weight"] = "linear_attention.decay_down.weight",
        ["self_attn.f_b_proj.weight"] = "linear_attention.decay_up.weight",
        ["self_attn.f_proj.weight"] = "linear_attention.decay.weight",
        ["self_attn.g_a_proj.weight"] = "linear_attention.gate_down.weight",
        ["self_attn.g_b_proj.weight"] = "linear_attention.gate_up.weight",
        ["self_attn.g_proj.weight"] = "linear_attention.gate.weight",
        ["self_attn.b_proj.weight"] = "linear_attention.beta.weight",
        ["self_attn.o_norm.weight"] = "linear_attention.norm.weight",
        ["self_attn.o_proj.weight"] = "linear_attention.out.weight",
    };

    /// <summary>
    /// Maps one tensor of a delta-attention layer; the q/k/v convolutions are collected into <paramref name="convolutions"/> (joined later by
    /// <see cref="AddJoinedConvolutions"/>). Returns false for tensors that are not delta-attention specific.
    /// </summary>
    private static bool TryMapDeltaLayer(HfConfig c, string prefix, string suffix, WeightInfo w, HfWeightCollector collector,
        SortedDictionary<string, WeightInfo?[]> convolutions)
    {
        switch (suffix)
        {
            case "self_attn.q_conv1d.weight" or "self_attn.k_conv1d.weight" or "self_attn.v_conv1d.weight":
                if (!convolutions.TryGetValue(prefix, out var parts))
                {
                    convolutions[prefix] = parts = new WeightInfo?[3];
                }

                parts[suffix[10] switch { 'q' => 0, 'k' => 1, _ => 2 }] = w;
                return true;
            case "self_attn.dt_bias":
                collector.Add(HfTransforms.ToF32(w, prefix + "linear_attention.dt_bias", new Shape(w.Shape.Elements())));
                return true;
            case "self_attn.A_log":
                collector.Add(HfTransforms.ToF32(w, prefix + "linear_attention.a", new Shape(LinearHeads(c)), static v => -MathF.Exp(v)));
                return true;
            default:
                if (s_delta.TryGetValue(suffix, out var canonical))
                {
                    collector.Add(HfTransforms.Plain(w, prefix + canonical));
                    return true;
                }

                return false;
        }
    }

    /// <summary>The joined short-convolution weight [3·heads·D, kernel] f32 of every delta-attention layer.</summary>
    /// <exception cref="InvalidDataException">A layer lacks one of its q / k / v convolutions.</exception>
    private static void AddJoinedConvolutions(SortedDictionary<string, WeightInfo?[]> convolutions, HfWeightCollector collector)
    {
        foreach (var (prefix, parts) in convolutions)
        {
            if (parts.Any(p => p is null))
            {
                throw new InvalidDataException($"{prefix}: delta attention needs q, k and v convolution weights.");
            }

            long channels = parts.Sum(p => p!.Shape[0].Fixed);
            long kernel = parts[0]!.Shape[-1].Fixed;
            var sources = parts.Select(p => p!).ToArray();
            collector.Add(WeightInfo.Derived(prefix + "linear_attention.conv1d.weight", DataType.F32, new Shape(channels, kernel), () =>
            {
                var joined = sources.SelectMany(p => HfTransforms.DecodeF32(p)).ToArray();
                return HfTransforms.EncodeF32(joined);
            }));
        }
    }
}
