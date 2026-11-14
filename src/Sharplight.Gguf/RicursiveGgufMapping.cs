namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>ricursive</c>: scaled embeddings that also feed a per-layer value residual (the value projection reads
/// x0 + (norm(x) − x0) · sigmoid(<c>attn_vres_lambda</c>)), per-head q/k norms, the rotary embedding on the sliding-window layers only
/// (default pattern: every fifth layer global and NoPE), a per-channel sigmoid output gate (<c>attn_gate</c>), sandwich norms around both
/// sub-layers and a SwiGLU whose gate projection is capped and up projection clamped per layer (<c>swiglu_clamp</c>).
/// </summary>
internal sealed class RicursiveGgufMapping : GgufDecoderFamilyMapping
{
    private static readonly Dictionary<string, string> s_tensors = new(StringComparer.Ordinal)
    {
        ["attn_gate.weight"] = "attention.gate.weight",
        ["attn_vres_lambda.weight"] = "attention.value_residual.weight",
    };

    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["ricursive"];

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string>? LayerTensorOverrides => s_tensors;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        string prefix = GgufDecoderHyperParameters.Prefix;
        int layerCount = g.Int("block_count");
        int[] sliding = GgufMappingHelpers.SlidingLayers(g, layerCount, 5, denseFirst: false);
        md[prefix + "attention.sliding_window"] = g.Int("attention.sliding_window");
        md[prefix + "attention.sliding_layers"] = sliding;
        md[prefix + "rope.layers"] = sliding;
        md[prefix + "rope.local_base"] = g.Float("rope.freq_base_swa", g.Float("rope.freq_base", GgufMappingHelpers.DefaultRopeBase));
        if (g.Float("embedding_scale", 0f) is var scale && scale != 0f)
        {
            md[prefix + "embedding_scale"] = scale;
        }

        md[prefix + "mlp.swiglu_clamp"] = GgufMappingHelpers.PerLayerFloat(g, "swiglu_clamp", layerCount);
        md[prefix + "mlp.swiglu_clamp_input"] = true;
    }
}
