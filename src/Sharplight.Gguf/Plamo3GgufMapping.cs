namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>plamo3</c> (PLaMo 3): sandwich RMS norms around both sub-layers (offsets already folded into the weights by the converter), per-head q/k
/// norms before the rotary embedding, a fused q/k/v projection, a fused gate+up SwiGLU MLP and, when the file has a window, sliding-window layers
/// (every 8th layer global by default) with their own rotary base (<c>rope.freq_base_swa</c>, default 10000).
/// </summary>
internal sealed class Plamo3GgufMapping : GgufDecoderFamilyMapping
{
    private const int DefaultPattern = 8;

    // llama.cpp loads the post norms without the ".weight" suffix.
    private static readonly Dictionary<string, string> s_tensors = new(StringComparer.Ordinal)
    {
        ["post_attention_norm"] = "attention.out_norm.weight",
        ["post_ffw_norm"] = "mlp.out_norm.weight",
    };

    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["plamo3"];

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string>? LayerTensorOverrides => s_tensors;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        if (g.Int("attention.sliding_window", 0) is not > 0 and var window)
        {
            return;
        }

        string prefix = GgufDecoderHyperParameters.Prefix;
        md[prefix + "attention.sliding_window"] = g.Int("attention.sliding_window");
        md[prefix + "attention.sliding_layers"] = GgufMappingHelpers.SlidingLayers(g, TrunkLayerCount(g), DefaultPattern, denseFirst: false);
        md[prefix + "rope.local_base"] = g.Float("rope.freq_base_swa", GgufMappingHelpers.DefaultRopeBase);
    }
}
