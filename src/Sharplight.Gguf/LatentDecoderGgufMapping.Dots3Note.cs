using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>dots3note</c> (dots.3): compressed-query latent attention with two geometries. The sliding-window layers
/// (<c>attention.sliding_window_pattern</c>, window <c>attention.sliding_window</c>, rope base <c>rope.freq_base_swa</c>) have their own head count
/// (per-layer <c>attention.head_count</c>), latent rank and head sizes (<c>attention.*_swa</c>); the full layers carry the DSA indexer. Both normalize
/// the shared rope key before rotating it (<c>attn_k_norm</c>) and gate every head's output by a sigmoid (<c>attn_gate</c>); sigmoid experts with a
/// selection bias and a shared expert follow dense first layers.
/// </summary>
internal sealed partial class LatentDecoderGgufMapping
{
    private static bool IsDots3Note(GgufMetadataView g) => g.GgufArchitecture == "dots3note";

    /// <summary>Whether layer <paramref name="il"/> is a sliding-window layer (<c>attention.sliding_window_pattern</c>).</summary>
    private static bool IsSlidingLayer(GgufMetadataView g, int il)
        => g.Raw("attention.sliding_window_pattern") is Array pattern && il < pattern.Length && Convert.ToBoolean(pattern.GetValue(il));

    /// <summary>The head count of the full layers (or of the sliding ones), from the scalar or per-layer <c>attention.head_count</c>.</summary>
    private static int HeadCount(GgufMetadataView g, bool sliding = false)
    {
        if (g.Raw("attention.head_count") is not Array heads)
        {
            return g.Int("attention.head_count");
        }

        for (int il = 0; il < TrunkLayers(g); il++)
        {
            if (IsSlidingLayer(g, il) == sliding)
            {
                return Convert.ToInt32(heads.GetValue(il));
            }
        }

        return Convert.ToInt32(heads.GetValue(0));
    }

    private static void AddDots3NoteHyperParameters(Dictionary<string, object> md, string a, GgufMetadataView g)
    {
        int layers = TrunkLayers(g);
        int ropeDims = g.Int("rope.dimension_count");
        int slidingKeyDims = g.Int("attention.key_length_mla_swa");
        md[a + "attention.sliding_layers"] = Enumerable.Range(0, layers).Select(il => IsSlidingLayer(g, il) ? 1 : 0).ToArray();
        md[a + "attention.sliding_window"] = g.Int("attention.sliding_window");
        md[a + "attention.sliding.head_count"] = HeadCount(g, sliding: true);
        md[a + "attention.sliding.latent_rank"] = g.Int("attention.kv_lora_rank_swa");
        md[a + "attention.sliding.nope_dim"] = slidingKeyDims - ropeDims;
        md[a + "attention.sliding.value_dim"] = g.Int("attention.value_length_mla_swa");
        // The same yarn correction as the full layers' scale (AttentionScale), on 1/sqrt of the sliding head size.
        md[a + "attention.sliding.scale"] = AttentionScale(g) * MathF.Sqrt(QueryKeyDims(g)) / MathF.Sqrt(slidingKeyDims);
        md[a + "rope.base_sliding"] = g.Float("rope.freq_base_swa", g.Float("rope.freq_base", GgufMappingHelpers.DefaultRopeBase));
    }

    /// <summary>Per-layer tensors of dots.3 beyond the shared latent-decoder ones, GGUF suffix to canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_dots3Note = new(StringComparer.Ordinal)
    {
        ["attn_k_norm.weight"] = "attention.k_rope_norm.weight",
        ["attn_gate.weight"] = "attention.gate.weight",
    };

    /// <summary>YaRN frequency divisors of the sliding-window layers, whose rope base differs (dots.3 scales both layer kinds).</summary>
    private static WeightInfo? SlidingYarnFactors(GgufMetadataView g)
    {
        if (!IsDots3Note(g) || g.Raw("rope.scaling.type") is not "yarn" || !g.Has("rope.freq_base_swa"))
        {
            return null;
        }

        var factors = RopeScaling.Yarn(
            g.Int("rope.dimension_count"),
            g.Float("rope.freq_base_swa"),
            g.Float("rope.scaling.factor"),
            g.Int("rope.scaling.original_context_length"),
            g.Float("rope.scaling.yarn_beta_fast", 32f),
            g.Float("rope.scaling.yarn_beta_slow", 1f)).Factors;
        return WeightInfo.Derived("rope.freq_factors_sliding", DataType.F32, new Shape(factors.Length), () =>
        {
            var bytes = new byte[factors.Length * 4];
            Buffer.BlockCopy(factors, 0, bytes, 0, bytes.Length);
            return bytes;
        });
    }
}
