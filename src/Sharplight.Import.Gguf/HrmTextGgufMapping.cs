namespace Sharplight.Import.Gguf;

/// <summary>
/// llama.cpp <c>hrm_text</c> (HRM-Text): blocks [0, S) hold the low stack and [S, 2S) the high stack (<c>hrm.layers_per_stack</c> = S), run in
/// <c>hrm.h_cycles</c> high cycles of <c>hrm.l_cycles</c> low passes plus one high pass, starting from the learned low state <c>hrm.z_l_init</c>
/// (canonical <c>cycles.*</c>). <c>block_count</c> is the expanded cache-slot count; the canonical layer count is 2S. Weight-less RMS norms,
/// a per-channel sigmoid attention gate (<c>attn_gate</c>) and scaled embeddings. <c>hrm.prefix_lm</c> is ignored (causal attention).
/// </summary>
internal sealed class HrmTextGgufMapping : GgufDecoderFamilyMapping
{
    private static readonly Dictionary<string, string> s_layerTensors = new(StringComparer.Ordinal)
    {
        ["attn_gate.weight"] = "attention.gate.weight",
    };

    private static readonly Dictionary<string, string> s_globalTensors = new(StringComparer.Ordinal)
    {
        ["hrm.z_l_init"] = "cycles.low_init",
    };

    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["hrm_text"];

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string>? LayerTensorOverrides => s_layerTensors;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string>? GlobalTensorOverrides => s_globalTensors;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        string prefix = GgufDecoderHyperParameters.Prefix;
        int stackLayers = g.Int("hrm.layers_per_stack");
        md[prefix + "layer_count"] = 2 * stackLayers;
        md[prefix + "cycles.stack_layers"] = stackLayers;
        md[prefix + "cycles.high"] = g.Int("hrm.h_cycles");
        md[prefix + "cycles.low"] = g.Int("hrm.l_cycles");
        if (g.Float("embedding_scale", 0f) is var scale && scale != 0f)
        {
            md[prefix + "embedding_scale"] = scale;
        }
    }
}
