namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>falcon</c> (Falcon 7B/40B/180B): multi-query / grouped attention with a fused q/k/v tensor, layer norm with bias, parallel residual,
/// plain GELU MLP. The 40B layout has a second norm (<c>attn_norm_2</c>) in front of the MLP; the 7B layout shares the attention norm.
/// </summary>
internal sealed class FalconGgufMapping : GgufDecoderFamilyMapping
{
    private static readonly Dictionary<string, string> s_tensors = new(StringComparer.Ordinal)
    {
        ["attn_norm_2.weight"] = "post_attention_norm.weight",
        ["attn_norm_2.bias"] = "post_attention_norm.bias",
    };

    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["falcon"];

    /// <inheritdoc/>
    protected override bool LayerNorm => true;

    /// <inheritdoc/>
    protected override bool Gated => false;

    /// <inheritdoc/>
    protected override string Activation => "gelu_tanh";

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string>? LayerTensorOverrides => s_tensors;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        md[GgufDecoderHyperParameters.Prefix + "residual.parallel"] = true;
    }
}
