using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>Hugging Face <c>internlm2</c> (InternLM2/2.5): a Llama decoder with a fused, per-key/value-head grouped <c>wqkv</c> and <c>feed_forward.w1/w2/w3</c> (gate, down, up).</summary>
internal sealed class InternLm2HfMapping : HfDecoderFamilyMapping
{
    private static readonly IReadOnlyDictionary<string, string> s_layerTensors = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["attention.wo.weight"] = "attention.o.weight",
        ["attention.wo.bias"] = "attention.o.bias",
        ["feed_forward.w1.weight"] = "mlp.gate.weight",
        ["feed_forward.w3.weight"] = "mlp.up.weight",
        ["feed_forward.w2.weight"] = "mlp.down.weight",
        ["attention_norm.weight"] = "input_norm.weight",
        ["ffn_norm.weight"] = "post_attention_norm.weight",
    };

    private static readonly IReadOnlyDictionary<string, string> s_globalTensors = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["model.tok_embeddings.weight"] = "token_embedding.weight",
        ["model.norm.weight"] = "output_norm.weight",
        ["output.weight"] = "lm_head.weight",
    };

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["internlm2"];

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> LayerTensors => s_layerTensors;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> GlobalTensors => s_globalTensors;

    /// <inheritdoc/>
    protected override bool TryMapSpecialLayerTensor(HfConfig c, string layerPrefix, string suffix, WeightInfo w, HfWeightCollector collector)
    {
        if (suffix is not ("attention.wqkv.weight" or "attention.wqkv.bias"))
        {
            return false;
        }

        int heads = HeadCount(c);
        HfFusedProjections.SplitGroupedQkv(w, layerPrefix, suffix.EndsWith(".bias", StringComparison.Ordinal) ? "bias" : "weight",
            heads, KvHeadCount(c, heads), HeadDimOf(c), collector);
        return true;
    }
}
