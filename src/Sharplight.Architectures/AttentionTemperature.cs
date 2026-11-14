using Sharplight.Graph;

namespace Sharplight.Architectures;

/// <summary>
/// Llama 4 / Mistral 3 / Mistral 4 attention temperature ("temperature tuning"): each query position is scaled by
/// <c>1 + scale · ln(1 + ⌊(pos + offset) / length⌋)</c>, which is 1 within the first <c>length</c> positions and grows slowly beyond; or, in the
/// power form (Beam), <c>(1 + (pos + offset) / length)^scale</c>.
/// </summary>
internal static class AttentionTemperature
{
    /// <summary>The temperature of every query row, shaped [T, 1, 1] to multiply [T, heads, dims] queries.</summary>
    /// <param name="b">Graph under construction.</param>
    /// <param name="positions">Positions [T, 1] (i32).</param>
    /// <param name="scale">β, the growth factor (not 0).</param>
    /// <param name="length">The position span of one step (positive).</param>
    /// <param name="offset">Offset added to the positions (0 for most models).</param>
    /// <param name="power">Whether to use the power form instead of the stepped logarithm.</param>
    public static Value Build(GraphBuilder b, Value positions, float scale, float length, float offset = 0f, bool power = false)
    {
        var pos = b.Emit1("Cast", [positions], Attributes.From(("type", "f32")));
        if (power)
        {
            var logBase = b.Unary("Log", b.Scale(pos, 1f / length, 1f + offset / length));
            return b.Reshape(b.Unary("Exp", b.Scale(logBase, scale)), 0, 1, 1);
        }

        var steps = b.Unary("Floor", b.Scale(pos, 1f / length, offset / length));
        var temperature = b.Scale(b.Unary("Log", b.Scale(steps, 1f, 1f)), scale, 1f);
        return b.Reshape(temperature, 0, 1, 1);
    }
}
