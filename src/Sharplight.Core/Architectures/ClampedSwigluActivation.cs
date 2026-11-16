using Sharplight.Graph;

namespace Sharplight.Architectures;

/// <summary>
/// SwiGLU with clamped operands (Step-3.5, Maple, Ricursive, DeepSeek-V4): the up projection is clamped to ±<c>limit</c> and the gate
/// side is limited to at most <c>limit</c>, either after the SiLU (<c>silu(gate)</c> capped, Step-3.5's dense and shared experts) or on
/// the gate projection before it (<c>silu(min(gate, limit))</c>, llama.cpp <c>swiglu_clamp</c>). The plain form is SiLU.
/// </summary>
internal sealed class ClampedSwigluActivation(float limit, bool clampGateInput) : IMlpActivation
{
    /// <inheritdoc/>
    public Value Activate(GraphBuilder b, Value x, string layerPrefix) => b.Silu(x);

    /// <inheritdoc/>
    public Value ActivateGated(GraphBuilder b, Value gate, Value up, string layerPrefix)
    {
        var activated = clampGateInput ? b.Silu(Clamp(b, gate, float.NegativeInfinity, limit)) : Clamp(b, b.Silu(gate), float.NegativeInfinity, limit);
        return b.Mul(activated, Clamp(b, up, -limit, limit));
    }

    private static Value Clamp(GraphBuilder b, Value x, float min, float max)
        => b.Emit1("Clip", [x], Attributes.From(("min", min), ("max", max)));
}
