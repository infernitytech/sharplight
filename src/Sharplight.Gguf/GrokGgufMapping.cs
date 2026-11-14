using Sharplight.Models;

namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>grok</c> (Grok-1 / Grok-2): scaled and soft-capped attention scores (<c>attention.output_scale</c>,
/// <c>attn_logit_softcapping</c>), sandwich norms (<c>attn_output_norm</c>, <c>layer_output_norm</c> / <c>post_ffw_norm</c>), softmax-routed
/// GELU experts with renormalized weights, an optional dense GELU MLP added to the experts with both scaled by √2/2 (mapped to the shared
/// expert of the MoE block), embedding and logit scales and an optional final logit soft cap. The defaults are llama.cpp's Grok-1 values.
/// </summary>
/// <remarks>
/// The router logit soft cap (<c>router_logit_softcapping</c>, default 30) is applied as in the reference implementation; llama.cpp reads it
/// but does not apply it.
/// </remarks>
internal sealed class GrokGgufMapping : GgufDecoderFamilyMapping
{
    private static readonly float s_branchScale = MathF.Sqrt(2f) / 2f;

    private static readonly Dictionary<string, string> s_tensors = new(StringComparer.Ordinal)
    {
        ["attn_output_norm.weight"] = "attention.out_norm.weight",
        ["layer_output_norm.weight"] = "mlp.out_norm.weight",
        ["ffn_gate.weight"] = "mlp.shared.gate.weight",
        ["ffn_up.weight"] = "mlp.shared.up.weight",
        ["ffn_down.weight"] = "mlp.shared.down.weight",
    };

    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["grok"];

    /// <inheritdoc/>
    protected override string Activation => "gelu_tanh";

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string>? LayerTensorOverrides => s_tensors;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        string prefix = GgufDecoderHyperParameters.Prefix;
        md[prefix + "attention.scale"] = g.Float("attention.output_scale", 0.08838834764831845f);
        md[prefix + "attention.logit_softcap"] = g.Float("attn_logit_softcapping", 30f);
        md[prefix + "embedding_scale"] = g.Float("embedding_scale", 78.38367176906169f);
        md[prefix + "logit_scale"] = g.Float("logit_scale", 0.5773502691896257f);
        md[prefix + "final_logit_softcap"] = g.Float("final_logit_softcapping", 0f);
        md[prefix + "moe.router_logit_softcap"] = g.Float("router_logit_softcapping", 30f);
    }

    /// <inheritdoc/>
    protected override void RefineFamilyHyperParameters(Dictionary<string, object> md, GgufMetadataView g, IReadOnlyDictionary<string, WeightInfo> weights)
    {
        // With the dense MLP (Grok-2) the layer output is (dense + experts) · √2/2.
        if (weights.Keys.Any(name => name.EndsWith(".mlp.shared.up.weight", StringComparison.Ordinal)))
        {
            string prefix = GgufDecoderHyperParameters.Prefix;
            float expertScale = md.TryGetValue(prefix + "moe.weights_scale", out var scale) ? Convert.ToSingle(scale) : 1f;
            md[prefix + "moe.weights_scale"] = expertScale * s_branchScale;
            md[prefix + "moe.shared_scale"] = s_branchScale;
        }
    }
}
