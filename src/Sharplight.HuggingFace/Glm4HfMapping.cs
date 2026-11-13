using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>glm4</c> (GLM-4-0414 dense): partial rotary embedding on interleaved pairs, a fused gate+up projection (gate first), and
/// sandwich norms — <c>post_self_attn_layernorm</c> and <c>post_mlp_layernorm</c> normalize the sub-layer outputs, <c>post_attention_layernorm</c> precedes the MLP.
/// </summary>
internal sealed class Glm4HfMapping : HfDecoderFamilyMapping
{
    private static readonly IReadOnlyDictionary<string, string> s_layerTensors = With(LlamaLayerTensors,
        ("post_self_attn_layernorm.weight", "attention.out_norm.weight"),
        ("post_mlp_layernorm.weight", "mlp.out_norm.weight"));

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["glm4"];

    /// <inheritdoc/>
    protected override string RopeStyle => "gptj";

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> LayerTensors => s_layerTensors;

    /// <inheritdoc/>
    protected override bool TryMapSpecialLayerTensor(HfConfig c, string layerPrefix, string suffix, WeightInfo w, HfWeightCollector collector)
    {
        if (suffix != "mlp.gate_up_proj.weight")
        {
            return false;
        }

        int ffn = c.Int("intermediate_size");
        collector.Add(HfTransforms.Plain(w.RowSlice("", 0, ffn), layerPrefix + "mlp.gate.weight"));
        collector.Add(HfTransforms.Plain(w.RowSlice("", ffn, ffn), layerPrefix + "mlp.up.weight"));
        return true;
    }
}
