using Sharplight.Graph;

namespace Sharplight.Architectures;

/// <summary>
/// Hyper-connections with per-token collapse and expansion weights (DeepSeek-V4's manifold-constrained mHC, Hunyuan v4): the residual
/// is N streams [T, N·H] (stream-major); before each block the streams are collapsed into its input with weights computed from all
/// streams, and the block output is expanded back into every stream. Pure graph code over standard ops, usable by any adapter.
/// </summary>
/// <remarks>
/// Weights below a module prefix: <c>fn.weight</c> [k, N·H], <c>base</c> [k] and <c>scale</c> [2 or 3] (f32), with k = 2N for a block
/// module (collapse | expansion), 2N + N² with a mixing matrix, and N for the final head (collapse only, <c>scale</c> [1]). With
/// m = fn · rms_norm(X) (unweighted, over all N·H values of a token):
/// <list type="bullet">
/// <item>collapse weights pre = sigmoid(m[0:N]·scale[0] + base[0:N]) + eps, block input Σ_s pre_s·X_s;</item>
/// <item>expansion weights post = magnitude·sigmoid(m[N:2N]·scale[1] + base[N:2N]) + post_offset;</item>
/// <item>optional mixing matrix C = sinkhorn(softmax_rows(m[2N:]·scale[2] + base[2N:]) + eps) (doubly stochastic, DeepSeek-V4);</item>
/// <item>after the block: X'_k = post_k·y + Σ_j C[j, k]·X_j (or + X_k without a mixing matrix).</item>
/// </list>
/// The final head collapses the streams with its own pre weights before the output norm, or averages them (<see cref="Mean"/>) when the
/// model has no head.
/// </remarks>
internal static class StreamHyperConnections
{
    /// <summary>Widens the embeddings [T, H] to the residual streams [T, N·H] (every stream starts as a copy).</summary>
    public static Value Expand(GraphBuilder b, Value x, int streams) => HyperConnections.Expand(b, x, streams);

    /// <summary>
    /// The input of the block whose module is below <paramref name="prefix"/>, collapsed from the streams x [T, N·H], and the
    /// expansion weights [T, N, 1] and optional mixing matrix [T, N, N] for <see cref="Merge"/>.
    /// </summary>
    public static (Value Input, Value Post, Value? Mixing) Collapse(GraphBuilder b, Value x, string prefix, StreamHyperConnectionOptions o, int hidden)
    {
        int n = o.Streams;
        var m = Mixes(b, x, prefix, o);
        var scale = b.Weight(prefix + "scale");
        var bias = b.Weight(prefix + "base");
        var pre = Weights(b, m, scale, bias, 0, n, 1f, o.Eps);
        var post = b.Reshape(Weights(b, m, scale, bias, 1, n, o.Magnitude, o.PostOffset), 0, n, 1);
        Value? mixing = null;
        if (o.SinkhornIterations > 0)
        {
            mixing = Sinkhorn(b, m, scale, bias, o);
        }

        return (Sum(b, x, pre, n, hidden, $"{prefix}input"), post, mixing);
    }

    /// <summary>The streams after a block: every stream gets the block output y [T, H] times its expansion weight, plus the (mixed) old streams.</summary>
    public static Value Merge(GraphBuilder b, Value x, Value y, Value post, Value? mixing, int streams, int hidden, string? name = null)
    {
        var streamsOf = b.Reshape(x, 0, streams, hidden);
        var expanded = b.Mul(post, b.Reshape(y, 0, 1, hidden));                                                          // [T, N, H]
        Value residual = streamsOf;
        if (mixing is not null)
        {
            // Σ_j C[j, k]·X_j: [T, j, k, 1] · [T, j, 1, H] summed over j.
            var products = b.Mul(b.Reshape(mixing, 0, streams, streams, 1), b.Reshape(x, 0, streams, 1, hidden));          // [T, j, k, H]
            residual = b.Reshape(b.Emit1("ReduceSum", [b.Permute(products, 0, 2, 3, 1)]), 0, streams, hidden);
        }

        return b.Reshape(b.Add(expanded, residual, name), 0, (long)streams * hidden);
    }

    /// <summary>The final collapse of the streams [T, N·H] into the hidden state [T, H] (module below <paramref name="prefix"/>).</summary>
    public static Value Head(GraphBuilder b, Value x, string prefix, StreamHyperConnectionOptions o, int hidden)
    {
        var m = Mixes(b, x, prefix, o);
        var pre = Weights(b, m, b.Weight(prefix + "scale"), b.Weight(prefix + "base"), 0, o.Streams, 1f, o.Eps);
        return Sum(b, x, pre, o.Streams, hidden, $"{prefix}output");
    }

    /// <summary>The unweighted mean of the streams [T, N·H] → [T, H] (a final collapse without a learned head).</summary>
    public static Value Mean(GraphBuilder b, Value x, int streams, int hidden)
        => b.Scale(b.Reshape(b.Emit1("ReduceSum", [b.Permute(b.Reshape(x, 0, streams, hidden), 0, 2, 1)]), 0, hidden), 1f / streams, name: "output_hc_mean");

    // fn · rms_norm(x): the norm is a per-token factor, so this equals fn · x scaled by the inverse RMS (Hunyuan v4's form).
    private static Value Mixes(GraphBuilder b, Value x, string prefix, StreamHyperConnectionOptions o)
        => b.Linear(b.RmsNorm(x, null, o.NormEps), prefix + "fn.weight");

    // Group g of N weights: factor · sigmoid(m[gN:(g+1)N] · scale[g] + base[gN:(g+1)N]) + offset, [T, N].
    private static Value Weights(GraphBuilder b, Value m, Value scale, Value bias, int g, int n, float factor, float offset)
    {
        var logits = b.Add(b.Mul(b.Slice(m, 1, (long)g * n, n), b.Slice(scale, 0, g, 1)), b.Slice(bias, 0, (long)g * n, n));
        return b.Scale(b.Sigmoid(logits), factor, offset);
    }

    // Σ_s w_s · X_s: x [T, N·H], w [T, N] → [T, H].
    private static Value Sum(GraphBuilder b, Value x, Value w, int n, int hidden, string name)
    {
        var weighted = b.Mul(b.Reshape(x, 0, n, hidden), b.Reshape(w, 0, n, 1), name);                                  // [T, N, H]
        return b.Reshape(b.Emit1("ReduceSum", [b.Permute(weighted, 0, 2, 1)]), 0, hidden);
    }

    // DeepSeek-V4: C = softmax_rows(logits) + eps, then columns, (rows, columns) × (iterations − 1) normalized to sum 1 (+ eps).
    private static Value Sinkhorn(GraphBuilder b, Value m, Value scale, Value bias, StreamHyperConnectionOptions o)
    {
        int n = o.Streams;
        var logits = b.Add(b.Mul(b.Reshape(b.Slice(m, 1, 2L * n, (long)n * n), 0, n, n), b.Slice(scale, 0, 2, 1)),
            b.Reshape(b.Slice(bias, 0, 2L * n, (long)n * n), n, n));
        var c = b.Scale(b.Emit1("Softmax", [logits]), 1f, o.Eps);
        c = NormalizeColumns(b, c, n, o.Eps);
        for (int i = 1; i < o.SinkhornIterations; i++)
        {
            c = b.Binary("Div", c, b.Scale(b.Emit1("ReduceSum", [c]), 1f, o.Eps));                                       // rows
            c = NormalizeColumns(b, c, n, o.Eps);
        }

        return c;
    }

    private static Value NormalizeColumns(GraphBuilder b, Value c, int n, float eps)
    {
        var columnSums = b.Reshape(b.Emit1("ReduceSum", [b.Permute(c, 0, 2, 1)]), 0, 1, n);                             // [T, 1, N]
        return b.Binary("Div", c, b.Scale(columnSums, 1f, eps));
    }
}
