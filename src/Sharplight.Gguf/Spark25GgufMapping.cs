namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>spark2_5</c> (Spark 2.5): RMS norms, sliding-window layers from a per-layer pattern with their own rotary base
/// (<c>rope.freq_base_swa</c>) and width (<c>rope.dimension_count_swa</c>), a per-head sigmoid output gate of the attention
/// (<c>attn_gate</c>, [heads, hidden]) and a GELU-gated MLP.
/// </summary>
internal sealed class Spark25GgufMapping : GgufDecoderFamilyMapping
{
    private static readonly Dictionary<string, string> s_tensors = new(StringComparer.Ordinal) { ["attn_gate.weight"] = "attention.gate.weight" };

    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["spark2_5"];

    /// <inheritdoc/>
    protected override string Activation => "gelu_tanh";

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string>? LayerTensorOverrides => s_tensors;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        string prefix = GgufDecoderHyperParameters.Prefix;
        md[prefix + "attention.sliding_window"] = g.Int("attention.sliding_window");
        md[prefix + "attention.sliding_layers"] = GgufMappingHelpers.SlidingLayers(g, TrunkLayerCount(g), 0, denseFirst: false);
        md[prefix + "rope.local_base"] = g.Float("rope.freq_base_swa", g.Float("rope.freq_base", GgufMappingHelpers.DefaultRopeBase));
        if (g.Has("rope.dimension_count_swa"))
        {
            md[prefix + "rope.local_dim"] = g.Int("rope.dimension_count_swa");
        }
    }
}
