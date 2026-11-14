using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>mistral-large4</c> (Mistral Large 4): latent attention without query compression, NoPE layers (<c>attention.rope_pattern</c>),
/// layers that attend an earlier layer's latent cache (<c>attention.kv_source_map</c>), SSMax factors on the attention and indexer queries
/// (<c>attn_ssmax</c>, <c>indexer.ssmax</c>), the DSA indexer with a LayerNorm'ed query input (<c>indexer.q_norm</c>) and shared indexer layers
/// (<c>attention.indexer.types</c>), learned rope frequencies (<c>rope_freqs</c>, LeRoPE) under YaRN without magnitude scaling, softmax experts with
/// a shared expert after dense first layers, and a final logit soft cap. llama.cpp rotates the indexer's queries and keys by a Hadamard matrix,
/// which leaves their dot products unchanged.
/// </summary>
internal sealed partial class LatentDecoderGgufMapping
{
    private static bool IsMistralLarge4(GgufMetadataView g) => g.GgufArchitecture == "mistral-large4";

    private static void AddMistralLarge4HyperParameters(Dictionary<string, object> md, string a, GgufMetadataView g)
    {
        int layers = TrunkLayers(g);
        md[a + "rope.layers"] = Enumerable.Range(0, layers)
            .Select(il => g.Raw("attention.rope_pattern") is Array flags ? (Convert.ToBoolean(flags.GetValue(il)) ? 1 : 0) : 1).ToArray();
        md[a + "attention.kv_source_layers"] = Enumerable.Range(0, layers)
            .Select(il => g.Raw("attention.kv_source_map") is Array sources ? Convert.ToInt32(sources.GetValue(il)) : il).ToArray();
        md[a + "final_logit_softcap"] = g.Float("final_logit_softcapping", 0f);
        // llama.cpp rotates the indexer with the model's rope type (interleaved pairs) and the learned frequencies; its norms use layer_norm_epsilon.
        md[a + "attention.indexer.rope_style"] = "gptj";
        md[a + "attention.indexer.norm_eps"] = g.Float("attention.layer_norm_epsilon", 1e-6f);
        if (g.Raw("rope.scaling.type") is "yarn")
        {
            // The rotated parts keep the magnitude yarn's attention factor leaves (1 for the log multiplier 1 the converter writes), and the
            // softmax scale stays 1/sqrt(head size).
            md[a + "rope.attention_scale"] = YarnMagnitude(g);
        }
    }

    /// <summary>Per-layer tensors of Mistral Large 4 beyond the shared latent-decoder ones, GGUF suffix to canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_mistralLarge4 = new(StringComparer.Ordinal)
    {
        ["attn_ssmax.weight"] = "attention.ssmax.weight",
        ["attn_ssmax.bias"] = "attention.ssmax.bias",
        ["indexer.ssmax.weight"] = "attention.indexer.ssmax.weight",
        ["indexer.ssmax.bias"] = "attention.indexer.ssmax.bias",
        ["indexer.q_norm.weight"] = "attention.indexer.q_norm.weight",
        ["indexer.q_norm.bias"] = "attention.indexer.q_norm.bias",
    };

    /// <summary>
    /// llama.cpp's cos/sin magnitude under YaRN for architectures other than DeepSeek-V2: its attention factor
    /// mscale(s, 1) / mscale(s, log multiplier) (mscale(s, 1) without a multiplier), divided by and then multiplied with the kernel's own 1 + 0.1·ln s.
    /// </summary>
    private static float YarnMagnitude(GgufMetadataView g)
    {
        float factor = g.Float("rope.scaling.factor", 1f);
        static float Mscale(float scale, float m) => scale <= 1f ? 1f : 0.1f * m * MathF.Log(scale) + 1f;
        float logMultiplier = g.Float("rope.scaling.yarn_log_multiplier", 0f);
        return logMultiplier != 0f ? Mscale(factor, 1f) / Mscale(factor, logMultiplier) : Mscale(factor, 1f);
    }

    /// <summary>The YaRN frequency divisors of the file's rope scaling (correction range from its beta keys).</summary>
    private static float[] YarnFactors(GgufMetadataView g)
        => RopeScaling.Yarn(
            g.Int("rope.dimension_count"),
            g.Float("rope.freq_base", GgufMappingHelpers.DefaultRopeBase),
            g.Float("rope.scaling.factor"),
            g.Int("rope.scaling.original_context_length"),
            g.Float("rope.scaling.yarn_beta_fast", 32f),
            g.Float("rope.scaling.yarn_beta_slow", 1f)).Factors;

    /// <summary>
    /// Learned frequency factors under YaRN (Mistral Large 4): llama.cpp divides θ by the stored factors and then blends with YaRN's ramp, so the
    /// canonical divisors are their product.
    /// </summary>
    private static WeightInfo YarnLearnedFactors(WeightInfo learned, GgufMetadataView g)
    {
        var yarn = YarnFactors(g);
        return WeightInfo.Derived(learned.Name, DataType.F32, new Shape(yarn.Length), () =>
        {
            var stored = learned.ReadAll();
            var bytes = new byte[yarn.Length * 4];
            for (int i = 0; i < yarn.Length; i++)
            {
                BitConverter.TryWriteBytes(bytes.AsSpan(i * 4, 4), BitConverter.ToSingle(stored, i * 4) * yarn[i]);
            }

            return bytes;
        });
    }
}
