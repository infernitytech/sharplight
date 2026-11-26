namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Hugging Face <c>domynedge</c> (Domyn Edge, remote code): sandwich norms around attention and MLP, an unweighted q norm and a weighted k norm per
/// head, rope only on the layers <c>position_layers</c> marks <c>"rope"</c> (the sliding-window layers of <c>layer_types</c>), per-layer head sizes
/// (<c>head_sizes</c>) and an attention scale of 1/sqrt(head size) per layer.
/// </summary>
internal sealed class DomynEdgeHfMapping : HfDecoderFamilyMapping
{
    private static readonly IReadOnlyDictionary<string, string> s_layerTensors = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["attention.q_proj.weight"] = "attention.q.weight",
        ["attention.k_proj.weight"] = "attention.k.weight",
        ["attention.v_proj.weight"] = "attention.v.weight",
        ["attention.o_proj.weight"] = "attention.o.weight",
        ["attention.q_proj.bias"] = "attention.q.bias",
        ["attention.k_proj.bias"] = "attention.k.bias",
        ["attention.v_proj.bias"] = "attention.v.bias",
        ["attention.o_proj.bias"] = "attention.o.bias",
        ["attention.k_norm.weight"] = "attention.k_norm.weight",
        ["feedforward.gate_proj.weight"] = "mlp.gate.weight",
        ["feedforward.up_proj.weight"] = "mlp.up.weight",
        ["feedforward.down_proj.weight"] = "mlp.down.weight",
        ["pre_attention_norm.weight"] = "input_norm.weight",
        ["pre_feedforward_norm.weight"] = "post_attention_norm.weight",
        ["post_attention_norm.weight"] = "attention.out_norm.weight",
        ["post_feedforward_norm.weight"] = "mlp.out_norm.weight",
    };

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["domynedge"];

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> LayerTensors => s_layerTensors;

    /// <inheritdoc/>
    public override IEnumerable<string> Validate(HfConfig config)
    {
        int layers = config.Int("num_hidden_layers");
        if (IntArray(config, "head_sizes") is not { } heads || heads.Length != layers)
        {
            yield return "domynedge: head_sizes must have num_hidden_layers entries.";
        }

        if (config.GetStringArray("position_layers") is not { } positions || positions.Length != layers)
        {
            yield return "domynedge: position_layers must have num_hidden_layers entries.";
        }
    }

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        int[] headSizes = IntArray(c, "head_sizes") ?? throw new InvalidDataException("domynedge: head_sizes missing.");
        var positions = c.GetStringArray("position_layers") ?? throw new InvalidDataException("domynedge: position_layers missing.");
        int[] rope = positions.Select(kind => kind == "rope" ? 1 : 0).ToArray();
        int ropeDim = headSizes.Where((_, il) => rope[il] != 0).DefaultIfEmpty(headSizes[0]).First();
        md["decoder.attention.head_dim"] = ropeDim;
        md["decoder.attention.layer_head_dims"] = headSizes;
        md["decoder.attention.layer_scales"] = headSizes.Select(size => 1f / MathF.Sqrt(size)).ToArray();
        md["decoder.attention.qk_norm_unweighted"] = true;
        md["decoder.rope.layers"] = rope;
        md["decoder.rope.dim"] = ropeDim;
        md["decoder.rope.local_dim"] = ropeDim;
        AddSlidingLayers(md, c);
    }
}
