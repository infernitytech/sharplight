namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>hunyuan_v1_moe</c> (Hunyuan-A13B; the remote-code checkpoints say <c>hunyuan</c>): the Hunyuan dense layer
/// (per-head q/k norms after the rotary embedding, NTK alpha folded into the rope base) with a softmax router
/// (<c>mlp.gate.wg</c>), renormalized top-k weights and an always-on shared MLP (<c>mlp.shared_mlp</c>).
/// </summary>
internal sealed class HunyuanMoeHfMapping : HfMoeFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["hunyuan_v1_moe", "hunyuan"];

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> LayerTensors => HunyuanDenseHfMapping.HunyuanLayerTensors;

    /// <inheritdoc/>
    protected override float RopeBase(HfConfig c) => HunyuanDenseHfMapping.AlphaRopeBase(c, base.RopeBase(c));

    /// <inheritdoc/>
    protected override void AddRopeScaling(Dictionary<string, object> md, HfConfig c)
    {
    }

    /// <inheritdoc/>
    protected override void AddRopeFactors(HfConfig c, HfWeightCollector collector)
    {
    }

    /// <inheritdoc/>
    protected override void AddFamilyHyperParameters(Dictionary<string, object> md, HfConfig c)
        => md["decoder.attention.qk_norm_after_rope"] = true;

    /// <inheritdoc/>
    /// <exception cref="NotSupportedException">Layers differ in expert count, top-k or expert size.</exception>
    protected override MoeRouting Routing(HfConfig c)
        => new(c.Int("num_experts"), Uniform(c, "moe_topk"), Uniform(c, "moe_intermediate_size") is > 0 and var size ? size : c.Int("intermediate_size"),
            "softmax", Normalize: true);

    // A per-layer list whose entries must agree (the decoder has one MoE configuration), or a plain number.
    private static int Uniform(HfConfig c, string key)
    {
        if (IntArray(c, key) is not { } values)
        {
            return c.GetInt(key) ?? 0;
        }

        if (values.Distinct().Count() != 1)
        {
            throw new NotSupportedException($"hunyuan: per-layer {key} values [{string.Join(", ", values)}] are not supported.");
        }

        return values[0];
    }
}
