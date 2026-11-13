namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>gpt-oss</c> (gpt-oss-20b / 120b): biased q/k/v/o, per-head attention sinks, alternating sliding-window (128) and global
/// layers that both use the YaRN-scaled rotary embedding, and a MoE block with a biased router, softmax over the top-4 logits,
/// per-expert biases and the clamped SwiGLU of gpt-oss (MXFP4 experts). llama.cpp's <c>post_attention_norm</c> is the norm in front
/// of the MLP.
/// </summary>
internal sealed class GptOssGgufMapping : GgufDecoderFamilyMapping
{
    private const int SlidingPattern = 2;

    private static readonly Dictionary<string, string> s_overrides = new(StringComparer.Ordinal)
    {
        ["post_attention_norm.weight"] = "post_attention_norm.weight",
    };

    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["gpt-oss"];

    /// <inheritdoc/>
    protected override string Activation => "swiglu_oai";

    /// <inheritdoc/>
    protected override bool NormalizesExpertWeights => false;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string>? LayerTensorOverrides => s_overrides;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        string prefix = GgufDecoderHyperParameters.Prefix;
        md[prefix + "moe.router_activation"] = "softmax_topk";
        md[prefix + "attention.sliding_window"] = g.Int("attention.sliding_window");
        md[prefix + "attention.sliding_layers"] = GgufMappingHelpers.SlidingLayers(g, TrunkLayerCount(g), SlidingPattern, denseFirst: false);
        md[prefix + "rope.local_base"] = g.Float("rope.freq_base_swa", g.Float("rope.freq_base", GgufMappingHelpers.DefaultRopeBase));
        md[prefix + "rope.local_freq_factors"] = true;
    }
}
