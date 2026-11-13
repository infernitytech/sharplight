using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>apertus</c> (Apertus): xIELU activation in a plain MLP, per-head q/k norms, <c>attention_layernorm</c> / <c>feedforward_layernorm</c> in front of the sub-layers. The activation
/// parameters (<c>act_fn.alpha_p</c>, <c>alpha_n</c>) are stored raw; the canonical weights hold the effective <c>αp = softplus(p)</c> and <c>αn = β + softplus(n)</c> with the
/// constants β = 0.5 and ε = −1e-6 of the reference implementation.
/// </summary>
internal sealed class ApertusHfMapping : HfDecoderFamilyMapping
{
    private const float Beta = 0.5f;
    private const float Epsilon = -1e-6f;

    private static readonly IReadOnlyDictionary<string, string> s_layerTensors = With(LlamaLayerTensors,
        ("mlp.gate_proj.weight", null),
        ("input_layernorm.weight", null),
        ("post_attention_layernorm.weight", null),
        ("attention_layernorm.weight", "input_norm.weight"),
        ("feedforward_layernorm.weight", "post_attention_norm.weight"));

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["apertus"];

    /// <inheritdoc/>
    protected override bool Gated => false;

    /// <inheritdoc/>
    protected override string Activation(HfConfig c) => "xielu";

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> LayerTensors => s_layerTensors;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        md["decoder.mlp.xielu_beta"] = Beta;
        md["decoder.mlp.xielu_eps"] = Epsilon;
    }

    /// <inheritdoc/>
    protected override bool TryMapSpecialLayerTensor(HfConfig c, string layerPrefix, string suffix, WeightInfo w, HfWeightCollector collector)
    {
        switch (suffix)
        {
            case "mlp.act_fn.alpha_p":
                collector.Add(Effective(w, layerPrefix + "mlp.act.alpha_p", raw => Softplus(raw)));
                return true;
            case "mlp.act_fn.alpha_n":
                collector.Add(Effective(w, layerPrefix + "mlp.act.alpha_n", raw => Beta + Softplus(raw)));
                return true;
            case "mlp.act_fn.beta" or "mlp.act_fn.eps":
                return true;   // constants of the reference implementation
            default:
                return false;
        }
    }

    private static float Softplus(float x) => x > 20f ? x : MathF.Log(1f + MathF.Exp(x));

    private static WeightInfo Effective(WeightInfo raw, string name, Func<float, float> transform)
        => WeightInfo.Derived(name, DataType.F32, new Shape(1), () => HfTransforms.EncodeF32([transform(HfTransforms.DecodeF32(raw)[0])]));
}
