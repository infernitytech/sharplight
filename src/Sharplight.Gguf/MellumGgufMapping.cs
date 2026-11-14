namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>mellum</c> (JetBrains Mellum 2): the Qwen3-MoE layer (per-head q/k norms, softmax router renormalized unless
/// <c>expert_weights_norm</c> says otherwise) with, when the file has a window, sliding-window layers (three of every four by default) that rotate with
/// <c>rope.freq_base_swa</c> (default: the global base). The NextN block llama.cpp appends is kept as the draft head.
/// </summary>
internal sealed class MellumGgufMapping : GgufDecoderFamilyMapping
{
    private const int DefaultPattern = 4;

    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["mellum"];

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        int window = g.Int("attention.sliding_window", 0);
        if (window <= 0)
        {
            return;
        }

        string prefix = GgufDecoderHyperParameters.Prefix;
        md[prefix + "attention.sliding_window"] = window;
        md[prefix + "attention.sliding_layers"] = GgufMappingHelpers.SlidingLayers(g, TrunkLayerCount(g), DefaultPattern, denseFirst: false);
        md[prefix + "rope.local_base"] = g.Float("rope.freq_base_swa", g.Float("rope.freq_base", GgufMappingHelpers.DefaultRopeBase));
    }
}
