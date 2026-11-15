namespace Sharplight.Import.Gguf;

/// <summary>
/// llama.cpp <c>deepseek2-ocr</c> (the language model of DeepSeek-OCR, DeepSeek-OCR-2 and Unlimited-OCR): DeepSeek-V2's layer without latent
/// attention, i.e. plain multi-head attention over q/k/v/o with neox rotary over the whole head (the file's <c>rope.dimension_count</c>, the
/// converter's <c>qk_rope_head_dim</c> of 0, does not apply), leading dense SwiGLU layers (the weights tell them apart), then softmax-routed SwiGLU
/// experts (llama.cpp's default when the file names no gating function), not renormalized unless <c>expert_weights_norm</c>, scaled by
/// <c>expert_weights_scale</c>, with an optional selection bias, plus always-on shared experts. Unlimited-OCR's sliding window is metadata only.
/// </summary>
internal sealed class Deepseek2OcrGgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["deepseek2-ocr"];

    /// <inheritdoc/>
    protected override bool NormalizesExpertWeights => false;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        md[GgufDecoderHyperParameters.Prefix + "rope.dim"] = md[GgufDecoderHyperParameters.Prefix + "attention.head_dim"];
        if (g.Int("expert_count", 0) > 0 && g.Int("expert_gating_func", 0) == 0)
        {
            md[GgufDecoderHyperParameters.Prefix + "moe.router_activation"] = "softmax";
        }
    }
}
