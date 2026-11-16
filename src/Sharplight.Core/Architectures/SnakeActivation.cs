using Sharplight.Graph;

namespace Sharplight.Architectures;

/// <summary>The snake activation of neural audio vocoders and codecs, <c>x + sin²(a·x) / a</c> with a per-channel <c>a</c>.</summary>
internal static class SnakeActivation
{
    /// <summary>Emits the activation over a time-major signal <c>[T, C]</c>.</summary>
    /// <param name="b">The graph builder.</param>
    /// <param name="x">The signal.</param>
    /// <param name="alpha">The weight <c>[C]</c> with <c>a</c>.</param>
    /// <param name="epsilon">Added to <c>a</c> in the reciprocal (SNAC: 1e-9; the iSTFTNet generator: 0).</param>
    public static Value Apply(GraphBuilder b, Value x, Value alpha, float epsilon = 0f)
    {
        var inverse = b.Unary("Recip", epsilon == 0f ? alpha : b.Scale(alpha, 1f, epsilon));
        return b.Add(x, b.Mul(inverse, b.Unary("Square", b.Unary("Sin", b.Mul(alpha, x)))));
    }
}
