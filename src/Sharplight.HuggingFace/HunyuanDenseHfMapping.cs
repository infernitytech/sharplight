namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>hunyuan_v1_dense</c> (Hunyuan 0.5B-7B): per-head q/k RMS norms (<c>query_layernorm</c> / <c>key_layernorm</c>) applied after the rotary
/// embedding, and NTK-aware "dynamic" rope scaling folded into the base (<c>base · alpha^(dim / (dim − 2))</c>).
/// </summary>
internal sealed class HunyuanDenseHfMapping : HfDecoderFamilyMapping
{
    private static readonly IReadOnlyDictionary<string, string> s_layerTensors = With(LlamaLayerTensors,
        ("self_attn.query_layernorm.weight", "attention.q_norm.weight"),
        ("self_attn.key_layernorm.weight", "attention.k_norm.weight"));

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["hunyuan_v1_dense"];

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> LayerTensors => s_layerTensors;

    /// <inheritdoc/>
    protected override float RopeBase(HfConfig c) => AlphaRopeBase(c, base.RopeBase(c));

    /// <summary>The q/k norm names of Hunyuan layers on top of the Llama names.</summary>
    internal static IReadOnlyDictionary<string, string> HunyuanLayerTensors => s_layerTensors;

    /// <summary>The rope base with Hunyuan's NTK-aware alpha (<c>rope_scaling.alpha</c>) folded in: <c>base · alpha^(dim / (dim − 2))</c>.</summary>
    internal static float AlphaRopeBase(HfConfig c, float ropeBase)
    {
        double alpha = HfMappingHelpers.NumberOr(c.RopeParameters, "alpha", 0);
        if (alpha <= 0)
        {
            return ropeBase;
        }

        int headDim = DecoderHfMapping.HeadDim(c);
        return ropeBase * (float)Math.Pow(alpha, headDim / (double)(headDim - 2));
    }

    /// <inheritdoc/>
    protected override void AddRopeScaling(Dictionary<string, object> md, HfConfig c)
    {
    }

    /// <inheritdoc/>
    protected override void AddRopeFactors(HfConfig c, HfWeightCollector collector)
    {
    }

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        md["decoder.attention.qk_norm_after_rope"] = true;
    }
}
