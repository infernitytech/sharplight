namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>smallthinker</c> (SmallThinker 4B-A0.6B / 21B-A3B): every layer is a MoE layer whose router reads the layer input before
/// the attention norm, with ReLU-gated experts and renormalized weights. Files with a sliding window (21B) use a 4096-token window on
/// every layer except each fourth one, which is a global NoPE layer.
/// </summary>
internal sealed class SmallThinkerGgufMapping : GgufDecoderFamilyMapping
{
    private const int SlidingPattern = 4;
    private const int Window = 4096;

    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["smallthinker"];

    /// <inheritdoc/>
    protected override string Activation => "relu";

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        string prefix = GgufDecoderHyperParameters.Prefix;
        md[prefix + "moe.router_input"] = "layer_input";
        md[prefix + "moe.normalize_weights"] = true;
        if (g.Int("attention.sliding_window", 0) <= 0)
        {
            return;
        }

        var sliding = GgufMappingHelpers.SlidingLayers(g, TrunkLayerCount(g), SlidingPattern, denseFirst: true);
        md[prefix + "attention.sliding_window"] = Window;
        md[prefix + "attention.sliding_layers"] = sliding;
        md[prefix + "rope.layers"] = Enumerable.Range(0, sliding.Length).Select(il => il % SlidingPattern != 0 ? 1 : 0).ToArray();
        md[prefix + "rope.local_base"] = g.Float("rope.freq_base_swa", g.Float("rope.freq_base", GgufMappingHelpers.DefaultRopeBase));
        md[prefix + "rope.local_freq_factors"] = true;
    }
}
