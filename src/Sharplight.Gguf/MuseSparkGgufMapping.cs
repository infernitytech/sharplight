using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>muse-spark</c> (Muse Spark): a weight-less norm of the embeddings, sliding-window layers from the per-layer pattern with rotary
/// embedding on their own base (interleaved pairs; the full-attention layers are NoPE), per-head q/k norms, a per-channel sigmoid output gate
/// after a weight-less norm of every head's output, a weight-less norm of the attention output, learned residual weights
/// (α · x + β · out) around both sub-layers, and in every layer softmax- or sigmoid-routed SwiGLU experts in a latent space with a norm of
/// the latent input and a norm of every expert output; the feed-forward output is scaled by its norm weight before it is normalized. Logits
/// are scaled, then soft-capped.
/// </summary>
internal sealed class MuseSparkGgufMapping : GgufDecoderFamilyMapping
{
    private static readonly Dictionary<string, string> s_tensors = new(StringComparer.Ordinal)
    {
        ["attn_gate.weight"] = "attention.gate.weight",
        ["attn_resid_alpha.weight"] = "attention.residual_alpha",
        ["attn_resid_beta.weight"] = "attention.residual_beta",
        ["ffn_resid_alpha.weight"] = "mlp.residual_alpha",
        ["ffn_resid_beta.weight"] = "mlp.residual_beta",
        ["ffn_pre_exp_norm.weight"] = "mlp.latent_in_norm.weight",
        ["ffn_post_exp_norm.weight"] = "mlp.experts.out_norm.weight",
    };

    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["muse-spark"];

    /// <inheritdoc/>
    protected override string RopeStyle => "gptj";

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string>? LayerTensorOverrides => s_tensors;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        string prefix = GgufDecoderHyperParameters.Prefix;
        int layerCount = g.Int("block_count");
        int[] sliding = GgufMappingHelpers.SlidingLayers(g, layerCount, 0, denseFirst: false);
        md[prefix + "attention.sliding_window"] = g.Int("attention.sliding_window");
        md[prefix + "attention.sliding_layers"] = sliding;
        md[prefix + "rope.layers"] = sliding;
        md[prefix + "rope.local_base"] = g.Float("rope.freq_base_swa", g.Float("rope.freq_base", GgufMappingHelpers.DefaultRopeBase));
        md[prefix + "post_norm_eps"] = g.Float("post_norm_epsilon", g.Float("attention.layer_norm_rms_epsilon"));
        md[prefix + "embedding_norm"] = true;
        md[prefix + "attention.out_head_norm"] = true;
        md[prefix + "post_norm.weight_first"] = true;
        md[prefix + "logit_scale"] = g.Float("logit_scale", 1f);
        if (g.Float("final_logit_softcapping", 0f) is > 0f and var cap)
        {
            md[prefix + "final_logit_softcap"] = cap;
        }
    }

    /// <summary>The attention output norm holds no weight: a weight of ones per layer (normalized after the multiplication, see post_norm.weight_first).</summary>
    protected override IEnumerable<WeightInfo> FamilySyntheticWeights(GgufMetadataView g, IReadOnlyDictionary<string, WeightInfo> weights)
    {
        int hidden = g.Int("embedding_length");
        for (int il = 0; il < g.Int("block_count"); il++)
        {
            yield return WeightInfo.Derived($"layers.{il}.attention.out_norm.weight", DataType.F32, new Shape(hidden), () =>
            {
                var bytes = new byte[hidden * 4];
                FillOnes(bytes);
                return bytes;
            });
        }
    }

    private static void FillOnes(byte[] bytes)
    {
        var floats = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(bytes);
        floats.Fill(1f);
    }
}
