using Sharplight.Models;

namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>stablelm</c> (StableLM 2): layer norm with bias, partial rotary, SwiGLU. The 12B model runs attention and MLP in parallel from one norm; llama.cpp tells the
/// layouts apart by the presence of <c>ffn_norm</c> (its <c>use_parallel_residual</c> key defaults to true when the source config lacks it), and so does this mapping.
/// </summary>
internal sealed class StableLmGgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["stablelm"];

    /// <inheritdoc/>
    protected override bool LayerNorm => true;

    /// <inheritdoc/>
    protected override void RefineFamilyHyperParameters(Dictionary<string, object> md, GgufMetadataView g, IReadOnlyDictionary<string, WeightInfo> weights)
    {
        md[GgufDecoderHyperParameters.Prefix + "residual.parallel"] = !weights.ContainsKey("layers.0.post_attention_norm.weight");
    }
}
