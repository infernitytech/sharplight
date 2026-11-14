namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>dbrx</c> (Databricks DBRX): layer norms without bias, a fused q/k/v clamped to ±<c>attention.clamp_kqv</c>, NeoX rotary embedding,
/// and softmax-routed SwiGLU experts with renormalized weights. The norm in front of the experts is stored as <c>attn_output_norm</c>.
/// </summary>
internal sealed class DbrxGgufMapping : GgufDecoderFamilyMapping
{
    private static readonly Dictionary<string, string> s_tensors = new(StringComparer.Ordinal)
    {
        ["attn_output_norm.weight"] = "post_attention_norm.weight",
    };

    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["dbrx"];

    /// <inheritdoc/>
    protected override bool LayerNorm => true;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string>? LayerTensorOverrides => s_tensors;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        if (g.Float("attention.clamp_kqv", 0f) is > 0f and var clamp)
        {
            md[GgufDecoderHyperParameters.Prefix + "attention.clip_qkv"] = clamp;
        }
    }
}
