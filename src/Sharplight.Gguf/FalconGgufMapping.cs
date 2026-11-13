using Sharplight.Models;

namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>falcon</c> (Falcon 7B/40B/180B): multi-query / grouped attention with a fused q/k/v tensor, layer norm with bias, parallel residual,
/// plain GELU MLP. The 40B layout has a second norm: llama.cpp stores Hugging Face's <c>ln_attn</c> as <c>attn_norm_2</c> (the attention input) and
/// <c>ln_mlp</c> as <c>attn_norm</c> (the MLP input); the 7B layout has only <c>attn_norm</c>, shared by both.
/// </summary>
internal sealed class FalconGgufMapping : GgufDecoderFamilyMapping, IGgufWeightTransformer
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

    /// <summary>Swaps the two norm names of every layer of the 40B layout, whose <c>attn_norm</c> feeds the MLP and <c>attn_norm_2</c> the attention.</summary>
    public void Transform(Dictionary<string, WeightInfo> weights, GgufMetadataView gguf)
    {
        foreach (string kind in new[] { "weight", "bias" })
        {
            string suffix = "post_attention_norm." + kind;
            foreach (string name in weights.Keys.Where(n => n.EndsWith("." + suffix, StringComparison.Ordinal)).ToList())
            {
                string inputName = name[..^suffix.Length] + "input_norm." + kind;
                (weights[name], weights[inputName]) = (weights[inputName].As(name), weights[name].As(inputName));
            }
        }
    }

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        md[GgufDecoderHyperParameters.Prefix + "residual.parallel"] = true;
    }
}
