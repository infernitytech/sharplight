namespace Sharplight.Gguf;

/// <summary>llama.cpp <c>smollm3</c> (SmolLM3): the llama graph where every fourth layer has no rotary embedding (NoPE).</summary>
internal sealed class SmolLm3GgufMapping : GgufDecoderFamilyMapping
{
    private const int NoRopeLayerStep = 4;

    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["smollm3"];

    /// <inheritdoc/>
    protected override bool PermutesQk => true;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        int layerCount = g.Int("block_count");
        md[GgufDecoderHyperParameters.Prefix + "rope.layers"] = Enumerable.Range(0, layerCount).Select(il => (il + 1) % NoRopeLayerStep != 0 ? 1 : 0).ToArray();
    }
}
