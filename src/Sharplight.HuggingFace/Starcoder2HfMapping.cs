namespace Sharplight.HuggingFace;

/// <summary>Hugging Face <c>starcoder2</c>: grouped-query attention with biases, layer norm with bias, plain biased GELU MLP (<c>c_fc</c>/<c>c_proj</c>), optional sliding window on every layer.</summary>
internal sealed class Starcoder2HfMapping : HfDecoderFamilyMapping
{
    private static readonly IReadOnlyDictionary<string, string> s_layerTensors = With(LlamaLayerTensors,
        ("mlp.c_fc.weight", "mlp.up.weight"),
        ("mlp.c_fc.bias", "mlp.up.bias"),
        ("mlp.c_proj.weight", "mlp.down.weight"),
        ("mlp.c_proj.bias", "mlp.down.bias"));

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["starcoder2"];

    /// <inheritdoc/>
    protected override bool LayerNorm => true;

    /// <inheritdoc/>
    protected override bool Gated => false;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> LayerTensors => s_layerTensors;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        if (c.GetInt("sliding_window") is > 0 and var window)
        {
            md["decoder.attention.sliding_window"] = window;
        }
    }
}
