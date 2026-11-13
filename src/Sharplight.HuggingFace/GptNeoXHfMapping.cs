using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>Hugging Face <c>gpt_neox</c> (Pythia, StableLM-alpha, RedPajama): fused per-head interleaved q/k/v, layer norm with bias, partial rotary, plain GELU MLP, optional parallel residual with two norms.</summary>
internal sealed partial class GptNeoXHfMapping : HfDecoderFamilyMapping
{
    private static readonly IReadOnlyDictionary<string, string> s_layerTensors = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["attention.dense.weight"] = "attention.o.weight",
        ["attention.dense.bias"] = "attention.o.bias",
        ["mlp.dense_h_to_4h.weight"] = "mlp.up.weight",
        ["mlp.dense_h_to_4h.bias"] = "mlp.up.bias",
        ["mlp.dense_4h_to_h.weight"] = "mlp.down.weight",
        ["mlp.dense_4h_to_h.bias"] = "mlp.down.bias",
        ["input_layernorm.weight"] = "input_norm.weight",
        ["input_layernorm.bias"] = "input_norm.bias",
        ["post_attention_layernorm.weight"] = "post_attention_norm.weight",
        ["post_attention_layernorm.bias"] = "post_attention_norm.bias",
    };

    private static readonly IReadOnlyDictionary<string, string> s_globalTensors = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["gpt_neox.embed_in.weight"] = "token_embedding.weight",
        ["gpt_neox.final_layer_norm.weight"] = "output_norm.weight",
        ["gpt_neox.final_layer_norm.bias"] = "output_norm.bias",
        ["embed_out.weight"] = "lm_head.weight",
    };

    [GeneratedRegex(@"^gpt_neox\.layers\.(\d+)\.(.+)$")]
    private static partial Regex NeoXLayerName();

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["gpt_neox"];

    /// <inheritdoc/>
    protected override bool LayerNorm => true;

    /// <inheritdoc/>
    protected override bool Gated => false;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> LayerTensors => s_layerTensors;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> GlobalTensors => s_globalTensors;

    /// <inheritdoc/>
    protected override Regex LayerName => NeoXLayerName();

    /// <inheritdoc/>
    protected override int RopeDimension(HfConfig c, int headDim)
        => (int)(headDim * (c.RopeDouble("partial_rotary_factor", "partial_rotary_factor", "rotary_pct") ?? 0.25));

    /// <inheritdoc/>
    protected override bool IsIgnored(string name)
        => base.IsIgnored(name) || name.EndsWith(".attention.bias", StringComparison.Ordinal) || name.EndsWith(".attention.masked_bias", StringComparison.Ordinal);

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        md["decoder.residual.parallel"] = c.GetBool("use_parallel_residual") ?? true;
    }

    /// <inheritdoc/>
    protected override bool TryMapSpecialLayerTensor(HfConfig c, string layerPrefix, string suffix, WeightInfo w, HfWeightCollector collector)
    {
        if (suffix is not ("attention.query_key_value.weight" or "attention.query_key_value.bias"))
        {
            return false;
        }

        int heads = HeadCount(c);
        HfFusedProjections.SplitGroupedQkv(w, layerPrefix, suffix.EndsWith(".bias", StringComparison.Ordinal) ? "bias" : "weight", heads, heads, HeadDimOf(c), collector);
        return true;
    }
}
