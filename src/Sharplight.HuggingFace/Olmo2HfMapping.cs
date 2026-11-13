namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>olmo2</c> and <c>olmo3</c>: norms only after the attention and MLP outputs (no norms in front), full-width q/k norms and, for OLMo 3,
/// sliding-window layers described by <c>layer_types</c>; rope scaling applies to the global layers only.
/// </summary>
internal sealed class Olmo2HfMapping : HfDecoderFamilyMapping
{
    private static readonly IReadOnlyDictionary<string, string> s_layerTensors = With(LlamaLayerTensors,
        ("input_layernorm.weight", null),
        ("post_attention_layernorm.weight", "attention.out_norm.weight"),
        ("post_feedforward_layernorm.weight", "mlp.out_norm.weight"));

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["olmo2", "olmo3"];

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> LayerTensors => s_layerTensors;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        md["decoder.pre_norm"] = false;
        AddSlidingLayers(md, c);
    }
}
