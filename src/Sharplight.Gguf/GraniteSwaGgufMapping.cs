namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>granite_swa</c> (Granite with sliding-window layers, dense or MoE with a shared expert): Granite's muP multipliers, per-head attention sinks,
/// sliding layers from the per-layer pattern and NoPE layers from <c>attention.rope_pattern</c>. llama.cpp rotates every layer with the global base (it
/// ignores per-layer thetas), and so does this mapping. The shared expert's fused gate+up tensor is split by the shared family mapping.
/// </summary>
internal sealed class GraniteSwaGgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["granite_swa"];

    /// <inheritdoc/>
    protected override bool PermutesQk => true;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        string prefix = GgufDecoderHyperParameters.Prefix;
        GgufMupScales.Add(md, g);
        md[prefix + "attention.sliding_window"] = g.Int("attention.sliding_window");
        md[prefix + "attention.sliding_layers"] = GgufMappingHelpers.SlidingLayers(g, TrunkLayerCount(g), 0, denseFirst: false);
        if (g.Raw("attention.rope_pattern") is Array ropeFlags)
        {
            md[prefix + "rope.layers"] = ropeFlags.Cast<object>().Select(flag => Convert.ToBoolean(flag) ? 1 : 0).ToArray();
        }
    }
}
