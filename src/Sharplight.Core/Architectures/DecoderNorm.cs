using Sharplight.Graph;

namespace Sharplight.Architectures;

/// <summary>
/// Norm factory of the decoder graph: the model's normalization kind (<c>norm.type</c>) turns a norm weight name into the
/// graph nodes. RMS norm multiplies by <c>name</c> and is non-parametric when the model has no such weight (Talkie); layer norm (mean-centred) multiplies by <c>name</c> and adds the
/// sibling <c>name</c> with <c>.weight</c> replaced by <c>.bias</c> when present, and is non-parametric (OLMo 1) when the weight is absent.
/// A grouped RMS norm (<c>norm.groups</c> G &gt; 1, K2-Horizon) normalizes G equal slices of the last axis separately and then multiplies by the
/// full-width weight.
/// </summary>
internal sealed class DecoderNorm
{
    private const string WeightSuffix = ".weight";
    private const string BiasSuffix = ".bias";

    private readonly bool _centred;
    private readonly int _groups;

    /// <summary>Creates the factory for a <c>norm.type</c> of "rms" or "layer" with <paramref name="groups"/> RMS norm groups.</summary>
    /// <exception cref="NotSupportedException">The type is neither, or a layer norm is grouped.</exception>
    public DecoderNorm(string type, int groups = 1)
    {
        _groups = groups;
        _centred = type switch
        {
            "rms" => false,
            "layer" => true,
            _ => throw new NotSupportedException($"decoder: unknown norm.type '{type}' (rms, layer)."),
        };
        if (_centred && groups > 1)
        {
            throw new NotSupportedException("decoder: norm.groups applies to RMS norms only.");
        }
    }

    /// <summary>Normalizes the last axis of <paramref name="x"/> with the weight (and bias) called <paramref name="weightName"/>.</summary>
    public Value Apply(GraphBuilder b, Value x, string weightName, float eps, string? name = null)
    {
        if (!_centred && _groups > 1)
        {
            return GroupedRmsNorm(b, x, b.TryWeight(weightName), eps, name);
        }

        if (!_centred)
        {
            return b.RmsNorm(x, b.TryWeight(weightName), eps, name);
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

    // RMS norm of G equal slices of the rows [T, H], then the full-width weight.
    private Value GroupedRmsNorm(GraphBuilder b, Value x, Value? weight, float eps, string? name)
    {
        long width = x.Shape[-1].Fixed;
        if (width % _groups != 0)
        {
            throw new InvalidDataException($"decoder: norm.groups {_groups} does not divide the hidden size {width}.");
        }

        var normed = b.Reshape(b.RmsNorm(b.Reshape(x, 0, _groups, width / _groups), null, eps), 0, width);
        return weight is null ? normed : b.Mul(normed, weight, name);
    }

    private static string BiasNameOf(string weightName)
        => weightName.EndsWith(WeightSuffix, StringComparison.Ordinal) ? weightName[..^WeightSuffix.Length] + BiasSuffix : weightName + BiasSuffix;
}
