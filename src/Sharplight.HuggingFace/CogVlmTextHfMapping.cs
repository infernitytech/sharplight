using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// The language model of CogVLM (remote-code <c>CogVLMForCausalLM</c>): Llama layers whose attention and MLP exist twice, as a language expert
/// (<c>self_attn.language_expert_query_key_value</c> / <c>language_expert_dense</c>, <c>mlp.language_mlp.*</c>) and a vision expert
/// (<c>vision_expert_*</c>, <c>mlp.vision_mlp.*</c>). The language expert becomes the layer's ordinary weights, the vision expert its media
/// weight set (<c>layers.{i}.media.*</c>, used by image rows); the fused q | k | v projections are split. Used by <see cref="CogVlmHfMapping"/>,
/// which separates the vision tower's tensors.
/// </summary>
internal sealed class CogVlmTextHfMapping : HfDecoderFamilyMapping
{
    private static readonly IReadOnlyDictionary<string, string> s_layerTensors = With(LlamaLayerTensors,
        ("self_attn.language_expert_dense.weight", "attention.o.weight"),
        ("self_attn.vision_expert_dense.weight", "media.attention.o.weight"),
        ("mlp.language_mlp.gate_proj.weight", "mlp.gate.weight"),
        ("mlp.language_mlp.up_proj.weight", "mlp.up.weight"),
        ("mlp.language_mlp.down_proj.weight", "mlp.down.weight"),
        ("mlp.vision_mlp.gate_proj.weight", "media.mlp.gate.weight"),
        ("mlp.vision_mlp.up_proj.weight", "media.mlp.up.weight"),
        ("mlp.vision_mlp.down_proj.weight", "media.mlp.down.weight"));

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["cogvlm"];

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> LayerTensors => s_layerTensors;

    /// <inheritdoc/>
    protected override bool TryMapSpecialLayerTensor(HfConfig c, string layerPrefix, string suffix, WeightInfo w, HfWeightCollector collector)
    {
        string? set = suffix switch
        {
            "self_attn.language_expert_query_key_value.weight" => "",
            "self_attn.vision_expert_query_key_value.weight" => "media.",
            _ => null,
        };
        if (set is null)
        {
            return false;
        }

        // Rows q | k | v, each heads · head_dim (multi-head attention).
        int heads = HeadCount(c);
        int kvHeads = KvHeadCount(c, heads);
        int headDim = HeadDimOf(c);
        var plain = HfTransforms.Plain(w, layerPrefix + set + "attention.qkv.weight");
        collector.Add(plain.RowSlice($"{layerPrefix}{set}attention.q.weight", 0, heads * headDim));
        collector.Add(plain.RowSlice($"{layerPrefix}{set}attention.k.weight", heads * headDim, kvHeads * headDim));
        collector.Add(plain.RowSlice($"{layerPrefix}{set}attention.v.weight", (heads + kvHeads) * headDim, kvHeads * headDim));
        return true;
    }
}
