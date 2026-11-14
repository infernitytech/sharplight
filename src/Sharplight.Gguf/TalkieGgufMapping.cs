namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>talkie</c>: weight-less RMS norms everywhere (embeddings, both sub-layer inputs, output), the q/k norm after the rotary
/// embedding with a per-head scalar gain on q only (<c>attn_q_norm</c>, [heads, 1]) and an unweighted norm on k, a learned per-layer skip
/// from the normed embeddings (<c>layer_output_scale</c>, x += scale · e0 after each layer) and a logit scale.
/// </summary>
internal sealed class TalkieGgufMapping : GgufDecoderFamilyMapping
{
    private static readonly Dictionary<string, string> s_tensors = new(StringComparer.Ordinal) { ["layer_output_scale.weight"] = "embedding_skip.weight" };

    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["talkie"];

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string>? LayerTensorOverrides => s_tensors;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        string prefix = GgufDecoderHyperParameters.Prefix;
        md[prefix + "embedding_norm"] = true;
        md[prefix + "attention.qk_norm_after_rope"] = true;
        md[prefix + "attention.qk_norm_unweighted"] = true;
        md[prefix + "logit_scale"] = g.Float("logit_scale", 1f);
    }
}
