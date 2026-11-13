using Sharplight.Graph;

namespace Sharplight.Architectures;

/// <summary>
/// Clamped SwiGLU of gpt-oss: <c>g = min(gate, limit)</c>, <c>u = clamp(up, −limit, limit)</c>, result <c>g·sigmoid(α·g)·(u + 1)</c>.
/// Only meaningful as a gated combination; the plain form is the gate part.
/// </summary>
internal sealed class SwigluOaiActivation(float alpha, float limit) : IMlpActivation
{
    /// <inheritdoc/>
    public Value Activate(GraphBuilder b, Value x, string layerPrefix)
    {
        var g = Clamp(b, x, float.NegativeInfinity, limit);
        return b.Mul(g, b.Sigmoid(b.Scale(g, alpha)));
    }

    /// <inheritdoc/>
    public Value ActivateGated(GraphBuilder b, Value gate, Value up, string layerPrefix)
    {
        var u = Clamp(b, up, -limit, limit);
        return b.Mul(Activate(b, gate, layerPrefix), b.Scale(u, 1f, 1f));
    }

    private static Value Clamp(GraphBuilder b, Value x, float min, float max)
        => b.Emit1("Clip", [x], Attributes.From(("min", min), ("max", max)));
}
