using Sharplight.Graph;

namespace Sharplight.Architectures;

/// <summary>
/// Gated hyper-connections: the residual stream is widened to <c>streams</c> copies of the hidden state ([T, streams·H],
/// stream-major) and every block reads a gated mix of the streams and writes its output back into each stream with a
/// per-stream weight. Replaces the pre-block norms and the plain residual add.
/// </summary>
/// <remarks>
/// Weights below a mixer prefix: <c>norm.weight</c> [streams·H] (RMS norm per stream, multiplied directly),
/// <c>down.weight</c> [rank, streams·H], <c>up.weight</c> [streams·H, rank] and, for block mixers,
/// <c>inject.weight</c> [streams, streams·H]. With Xn the per-stream normalized residual:
/// mix = mean over streams of sigmoid(up(silu(down(Xn) / streams))) ⊙ Xn; inject = 2·sigmoid(inject(Xn) / streams);
/// after the block X += inject_s · y for every stream s.
/// </remarks>
internal static class HyperConnections
{
    /// <summary>Widens the embeddings [T, H] to the residual streams [T, streams·H] (every stream starts as a copy).</summary>
    public static Value Expand(GraphBuilder b, Value x, int streams) =>
        b.Emit1("Tile", [x], Attributes.From(("repeats", new long[] { 1, streams })), "hc_expand");

    /// <summary>RMS-normalizes every stream of x [T, streams·H] and applies the weight [streams·H]; returns [T, streams, H].</summary>
    public static Value StreamNorm(GraphBuilder b, Value x, Value weight, int streams, int hidden, float eps) =>
        b.Mul(b.RmsNorm(b.Reshape(x, 0, streams, hidden), null, eps), b.Reshape(weight, streams, hidden));

    /// <summary>
    /// The block input mixed from the streams [T, H] and, when the mixer has an <c>inject.weight</c>, the per-stream output
    /// weights [T, streams, 1] for <see cref="Inject"/>.
    /// </summary>
    public static (Value Mixed, Value? Inject) Mix(GraphBuilder b, Value x, string prefix, int streams, int hidden, float eps)
    {
        var xn = StreamNorm(b, x, b.Weight(prefix + "norm.weight"), streams, hidden, eps);
        var flat = b.Reshape(xn, 0, (long)streams * hidden);
        var low = b.Silu(b.Scale(b.Linear(flat, prefix + "down.weight"), 1f / streams));
        var gate = b.Sigmoid(b.Linear(low, prefix + "up.weight"));
        // Mean over the streams: [T, streams, H] → [T, H, streams] → sum of the last axis.
        var gated = b.Mul(b.Reshape(gate, 0, streams, hidden), xn);
        var mixed = b.Scale(b.Reshape(b.Emit1("ReduceSum", [b.Permute(gated, 0, 2, 1)]), 0, hidden), 1f / streams, name: $"{prefix}mixed");
        if (b.TryWeight(prefix + "inject.weight") is null)
        {
            return (mixed, null);
        }

        var inject = b.Scale(b.Sigmoid(b.Scale(b.Linear(flat, prefix + "inject.weight"), 1f / streams)), 2f);
        return (mixed, b.Reshape(inject, 0, streams, 1));
    }

    /// <summary>Adds the block output y [T, H], weighted per stream by <paramref name="inject"/> [T, streams, 1], to every stream of x.</summary>
    public static Value Inject(GraphBuilder b, Value x, Value y, Value inject, int streams, int hidden, string? name = null)
    {
        var spread = b.Mul(b.Reshape(y, 0, 1, hidden), inject);
        return b.Add(x, b.Reshape(spread, 0, (long)streams * hidden), name);
    }
}
