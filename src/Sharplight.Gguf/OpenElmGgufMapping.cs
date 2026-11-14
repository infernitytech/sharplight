namespace Sharplight.Gguf;

/// <summary>llama.cpp <c>openelm</c> (OpenELM): per-layer query / key-value head counts and feed-forward sizes, a fused q/k/v tensor and per-head q/k norms.</summary>
internal sealed class OpenElmGgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["openelm"];

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        string prefix = GgufDecoderHyperParameters.Prefix;
        int layerCount = g.Int("block_count");
        md[prefix + "attention.layer_head_counts"] = GgufMappingHelpers.PerLayer(g, "attention.head_count", layerCount);
        md[prefix + "attention.layer_kv_head_counts"] = GgufMappingHelpers.PerLayer(g, "attention.head_count_kv", layerCount);
    }

    /// <inheritdoc/>
    protected override (int Heads, int KvHeads) HeadsOf(GgufMetadataView g, int layer)
        => (GgufMappingHelpers.PerLayer(g, "attention.head_count", g.Int("block_count"))[layer], GgufMappingHelpers.PerLayer(g, "attention.head_count_kv", g.Int("block_count"))[layer]);
}
