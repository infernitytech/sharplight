namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>exaone4</c> (EXAONE 4.0): norms only after the attention and MLP outputs (<c>post_attention_layernorm</c>, <c>post_feedforward_layernorm</c>), per-head q/k norms
/// and, when the config has a sliding window, interleaved sliding layers that are the only ones with a rotary embedding.
/// </summary>
internal sealed class Exaone4HfMapping : HfDecoderFamilyMapping
{
    private static readonly IReadOnlyDictionary<string, string> s_layerTensors = With(LlamaLayerTensors,
        ("input_layernorm.weight", null),
        ("post_attention_layernorm.weight", "attention.out_norm.weight"),
        ("post_feedforward_layernorm.weight", "mlp.out_norm.weight"));

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["exaone4"];

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> LayerTensors => s_layerTensors;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        md["decoder.pre_norm"] = false;
        if (AddSlidingLayers(md, c) is { } sliding)
        {
            md["decoder.rope.layers"] = sliding;
        }
    }
}
