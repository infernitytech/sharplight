namespace Sharplight.Import.Gguf;

/// <summary>
/// llama.cpp <c>domynedge</c> (Domyn Edge): sandwich norms, an unweighted q norm and a weighted k norm per head, rope only on the layers of
/// <c>attention.rope_pattern</c>, which are also the sliding-window layers (<c>attention.sliding_window_pattern</c>), with their own head size
/// (<c>attention.key_length_swa</c>) and the global rope base; attention scale 1/sqrt(head size) per layer.
/// </summary>
internal sealed class DomynEdgeGgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["domynedge"];

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        string prefix = GgufDecoderHyperParameters.Prefix;
        int layerCount = g.Int("block_count");
        int globalDim = g.Int("attention.key_length");
        int localDim = g.Int("attention.key_length_swa", globalDim);
        int[] sliding = Flags(g, "attention.sliding_window_pattern", layerCount);
        int[] rope = g.Raw("attention.rope_pattern") is Array ? Flags(g, "attention.rope_pattern", layerCount) : sliding;
        int[] headDims = sliding.Select(s => s != 0 ? localDim : globalDim).ToArray();
        int ropeDim = g.Int("rope.dimension_count_swa", localDim);
        md[prefix + "attention.head_dim"] = localDim;
        md[prefix + "attention.layer_head_dims"] = headDims;
        md[prefix + "attention.layer_scales"] = headDims.Select(size => 1f / MathF.Sqrt(size)).ToArray();
        md[prefix + "attention.qk_norm_unweighted"] = true;
        md[prefix + "attention.sliding_window"] = g.Int("attention.sliding_window");
        md[prefix + "attention.sliding_layers"] = sliding;
        md[prefix + "rope.layers"] = rope;
        md[prefix + "rope.dim"] = ropeDim;
        md[prefix + "rope.local_dim"] = ropeDim;
        md[prefix + "rope.local_base"] = g.Float("rope.freq_base", GgufMappingHelpers.DefaultRopeBase);
    }

    private static int[] Flags(GgufMetadataView g, string key, int layerCount)
    {
        var values = g.Raw(key) as Array ?? throw new InvalidDataException($"domynedge: {key} missing.");
        return Enumerable.Range(0, layerCount).Select(il => Convert.ToBoolean(values.GetValue(il)) ? 1 : 0).ToArray();
    }
}
