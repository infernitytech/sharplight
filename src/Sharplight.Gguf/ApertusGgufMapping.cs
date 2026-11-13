using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>apertus</c> (Apertus): xIELU activation in a plain MLP and per-head q/k norms. The activation parameters are stored as
/// per-layer metadata arrays of raw values; the canonical weights hold the effective <c>αp = softplus(p)</c> and <c>αn = β + softplus(n)</c>.
/// </summary>
internal sealed class ApertusGgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["apertus"];

    /// <inheritdoc/>
    protected override bool Gated => false;

    /// <inheritdoc/>
    protected override string Activation => "xielu";

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        string prefix = GgufDecoderHyperParameters.Prefix;
        md[prefix + "mlp.xielu_beta"] = Parameters(g, "xielu.beta")[0];
        md[prefix + "mlp.xielu_eps"] = Parameters(g, "xielu.eps")[0];
    }

    /// <inheritdoc/>
    protected override IEnumerable<WeightInfo> FamilySyntheticWeights(GgufMetadataView g)
    {
        var alphaPositive = Parameters(g, "xielu.alpha_p");
        var alphaNegative = Parameters(g, "xielu.alpha_n");
        var beta = Parameters(g, "xielu.beta");
        for (int layer = 0; layer < alphaPositive.Length; layer++)
        {
            yield return ScalarWeight($"layers.{layer}.mlp.act.alpha_p", Softplus(alphaPositive[layer]));
            yield return ScalarWeight($"layers.{layer}.mlp.act.alpha_n", beta[layer] + Softplus(alphaNegative[layer]));
        }
    }

    private static float[] Parameters(GgufMetadataView g, string key)
        => g.All.TryGetValue(key, out var raw) && raw is Array values ? values.Cast<object>().Select(Convert.ToSingle).ToArray() : throw new InvalidDataException($"GGUF metadata {key} (per-layer array) missing.");

    private static float Softplus(float x) => x > 20f ? x : MathF.Log(1f + MathF.Exp(x));

    private static WeightInfo ScalarWeight(string name, float value)
        => WeightInfo.Derived(name, DataType.F32, new Shape(1), () => BitConverter.GetBytes(value));
}
