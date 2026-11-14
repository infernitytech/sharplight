namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>deci</c> (Llama-3.1-Nemotron 51B / Ultra): per-layer query and key/value head counts and feed-forward sizes. A layer with no
/// query heads has no attention, one with query heads but no key/value heads replaces attention by a single linear map
/// (<c>attn_output</c> of the normed input), and a layer with no feed-forward size has no MLP. q/k are stored in permuted order.
/// </summary>
/// <remarks>
/// llama.cpp skips a layer without MLP entirely (its attention output is dropped as well); Sharplight keeps the attention residual as the
/// Hugging Face model does.
/// </remarks>
internal sealed class DeciGgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["deci"];

    /// <inheritdoc/>
    protected override bool PermutesQk => true;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        string prefix = GgufDecoderHyperParameters.Prefix;
        int layerCount = g.Int("block_count");
        md[prefix + "attention.layer_head_counts"] = GgufMappingHelpers.PerLayer(g, "attention.head_count", layerCount);
        md[prefix + "attention.layer_kv_head_counts"] = GgufMappingHelpers.PerLayer(g, "attention.head_count_kv", layerCount, GgufDecoderHyperParameters.LargestOf(g, "attention.head_count"));
    }

    /// <inheritdoc/>
    protected override (int Heads, int KvHeads) HeadsOf(GgufMetadataView g, int layer)
    {
        int layerCount = g.Int("block_count");
        return (GgufMappingHelpers.PerLayer(g, "attention.head_count", layerCount)[layer],
            GgufMappingHelpers.PerLayer(g, "attention.head_count_kv", layerCount, GgufDecoderHyperParameters.LargestOf(g, "attention.head_count"))[layer]);
    }
}
