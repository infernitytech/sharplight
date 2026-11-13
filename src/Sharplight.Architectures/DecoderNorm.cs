using Sharplight.Graph;

namespace Sharplight.Architectures;

/// <summary>
/// Norm factory of the decoder graph: the model's normalization kind (<c>norm.type</c>) turns a norm weight name into the
/// graph nodes. RMS norm multiplies by <c>name</c> (required); layer norm (mean-centred) multiplies by <c>name</c> and adds the
/// sibling <c>name</c> with <c>.weight</c> replaced by <c>.bias</c> when present, and is non-parametric (OLMo 1) when the weight is absent.
/// </summary>
internal sealed class DecoderNorm
{
    private const string WeightSuffix = ".weight";
    private const string BiasSuffix = ".bias";

    private readonly bool _centred;

    /// <summary>Creates the factory for a <c>norm.type</c> of "rms" or "layer".</summary>
    /// <exception cref="NotSupportedException">The type is neither.</exception>
    public DecoderNorm(string type)
    {
        _centred = type switch
        {
            "rms" => false,
            "layer" => true,
            _ => throw new NotSupportedException($"decoder: unknown norm.type '{type}' (rms, layer)."),
        };
    }

    /// <summary>Normalizes the last axis of <paramref name="x"/> with the weight (and bias) called <paramref name="weightName"/>.</summary>
    public Value Apply(GraphBuilder b, Value x, string weightName, float eps, string? name = null)
    {
        if (!_centred)
        {
            return b.RmsNorm(x, b.Weight(weightName), eps, name);
        }

        var weight = b.TryWeight(weightName);
        var bias = weight is null ? null : b.TryWeight(BiasNameOf(weightName));
        var operands = new List<Value> { x };
        if (weight is not null)
        {
            operands.Add(weight);
        }

        if (bias is not null)
        {
            operands.Add(bias);
        }

        return b.Emit1("LayerNorm", operands, Attributes.From(("eps", eps)), name);
    }

    private static string BiasNameOf(string weightName)
        => weightName.EndsWith(WeightSuffix, StringComparison.Ordinal) ? weightName[..^WeightSuffix.Length] + BiasSuffix : weightName + BiasSuffix;
}
