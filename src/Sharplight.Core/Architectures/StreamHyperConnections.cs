using Sharplight.Graph;
using Sharplight.Tensors;

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
/// <item>collapse weights pre = sigmoid(m[0:N]·scale[0] + base[0:N]) + pre_eps, block input Σ_s pre_s·X_s;</item>
/// <item>expansion weights post = magnitude·sigmoid(m[N:2N]·scale[1] + base[N:2N]) + post_offset;</item>
/// <item>optional mixing matrix C = sinkhorn(softmax_rows(clamp(m[2N:]·scale[2] + base[2N:])) + mixing_eps) (doubly stochastic,
/// DeepSeek-V4; Xing4.0 clamps the logits to ±30 and adds no mixing_eps);</item>
/// <item>or the Raon_mHC chart (<see cref="StreamHyperConnectionOptions.ChartMixing"/>, k = 2N + (N−1)², <c>scale</c> [4]): z = m[2N:]·scale[2] +
/// base[2N:] as an (N−1)×(N−1) matrix Z, E = Q·Z·Qᵀ with the Helmert basis Q [N, N−1] (orthonormal columns orthogonal to 1),
/// ρ = max(0, max_ij(−E_ij / B_ij)) with B = (I + 11ᵀ/N)/2, Φ = B + E/(1 + ρ) (doubly stochastic, non-negative) and
/// C = I + sigmoid(scale[3])·(Φ − I), applied transposed;</item>
/// <item>after the block: X'_k = post_k·y + Σ_j C[j, k]·X_j, or Σ_j C[k, j]·X_j when the matrix is transposed (Xing4.0), or + X_k without a
/// mixing matrix.</item>
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
        var (pre, post, mixing) = Weigh(b, x, prefix, o);
        return (Apply(b, x, pre, o, hidden, $"{prefix}input"), post, mixing);
    }

    /// <summary>
    /// The collapse weights [T, N], expansion weights [T, N, 1] and optional mixing matrix [T, N, N] of the module below <paramref name="prefix"/>,
    /// computed from the streams x [T, N·H] (<see cref="Collapse"/> applies the collapse weights to the same streams; with
    /// <see cref="StreamHyperConnectionOptions.LaggedCollapse"/> they collapse the input of the next block instead).
    /// </summary>
    public static (Value Pre, Value Post, Value? Mixing) Weigh(GraphBuilder b, Value x, string prefix, StreamHyperConnectionOptions o)
    {
        int n = o.Streams;
        var m = Mixes(b, x, prefix, o);
        var scale = b.Weight(prefix + "scale");
        var bias = b.Weight(prefix + "base");
        var pre = Weights(b, m, scale, bias, 0, n, 1f, o.PreEps);
        var post = b.Reshape(Weights(b, m, scale, bias, 1, n, o.Magnitude, o.PostOffset), 0, n, 1);
        Value? mixing = null;
        if (o.ChartMixing)
        {
            mixing = Chart(b, m, scale, bias, n);
        }
        else if (o.SinkhornIterations > 0)
        {
            mixing = Sinkhorn(b, m, scale, bias, o);
        }

        return (pre, post, mixing);
    }

    /// <summary>The streams x [T, N·H] collapsed with the weights <paramref name="pre"/> [T, N] into [T, H].</summary>
    public static Value Apply(GraphBuilder b, Value x, Value pre, StreamHyperConnectionOptions o, int hidden, string name)
        => Sum(b, x, pre, o.Streams, hidden, name);

    /// <summary>Stream <paramref name="stream"/> of x [T, N·H] as [T, H] (the collapse of the first block with lagged collapse weights).</summary>
    public static Value Stream(GraphBuilder b, Value x, int stream, int hidden) => b.Slice(x, 1, (long)stream * hidden, hidden);

    /// <summary>The streams after a block: every stream gets the block output y [T, H] times its expansion weight, plus the (mixed) old streams.</summary>
    public static Value Merge(GraphBuilder b, Value x, Value y, Value post, Value? mixing, StreamHyperConnectionOptions o, int hidden, string? name = null)
    {
        int streams = o.Streams;
        var streamsOf = b.Reshape(x, 0, streams, hidden);
        var expanded = b.Mul(post, b.Reshape(y, 0, 1, hidden));                                                          // [T, N, H]
        Value residual = streamsOf;
        if (mixing is not null)
        {
            // Σ_j C[j, k]·X_j: [T, j, k, 1] · [T, j, 1, H] summed over j (C transposed first for Σ_j C[k, j]·X_j).
            if (o.MixingTransposed)
            {
                mixing = b.Permute(mixing, 0, 2, 1);
            }

            var products = b.Mul(b.Reshape(mixing, 0, streams, streams, 1), b.Reshape(x, 0, streams, 1, hidden));          // [T, j, k, H]
            residual = b.Reshape(b.Emit1("ReduceSum", [b.Permute(products, 0, 2, 3, 1)]), 0, streams, hidden);
        }

        return b.Reshape(b.Add(expanded, residual, name), 0, (long)streams * hidden);
    }

    /// <summary>The final collapse of the streams [T, N·H] into the hidden state [T, H] (module below <paramref name="prefix"/>).</summary>
    public static Value Head(GraphBuilder b, Value x, string prefix, StreamHyperConnectionOptions o, int hidden)
    {
        var m = Mixes(b, x, prefix, o);
        var pre = Weights(b, m, b.Weight(prefix + "scale"), b.Weight(prefix + "base"), 0, o.Streams, 1f, o.PreEps);
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

    // DeepSeek-V4: C = softmax_rows(logits) + mixing_eps, then columns, (rows, columns) × (iterations − 1) normalized to sum 1 (+ eps).
    private static Value Sinkhorn(GraphBuilder b, Value m, Value scale, Value bias, StreamHyperConnectionOptions o)
    {
        int n = o.Streams;
        var logits = b.Add(b.Mul(b.Reshape(b.Slice(m, 1, 2L * n, (long)n * n), 0, n, n), b.Slice(scale, 0, 2, 1)),
            b.Reshape(b.Slice(bias, 0, 2L * n, (long)n * n), n, n));
        if (o.MixingClamp > 0f)
        {
            logits = b.Emit1("Clip", [logits], Attributes.From(("min", -o.MixingClamp), ("max", o.MixingClamp)));
        }

        var c = b.Emit1("Softmax", [logits]);
        if (o.MixingEps != 0f)
        {
            c = b.Scale(c, 1f, o.MixingEps);
        }

        c = NormalizeColumns(b, c, n, o.Eps);
        for (int i = 1; i < o.SinkhornIterations; i++)
        {
            c = b.Binary("Div", c, b.Scale(b.Emit1("ReduceSum", [c]), 1f, o.Eps));                                       // rows
            c = NormalizeColumns(b, c, n, o.Eps);
        }

        return c;
    }

    // Raon_mHC: the stream mixing matrix [T, N, N] from the Helmert chart of the (N − 1)² logits m[2N:] (see the remarks).
    private static Value Chart(GraphBuilder b, Value m, Value scale, Value bias, int n)
    {
        int nm = n - 1;
        int coordinates = nm * nm;
        int entries = n * n;
        var z = b.Add(b.Mul(b.Slice(m, 1, 2L * n, coordinates), b.Slice(scale, 0, 2, 1)), b.Slice(bias, 0, 2L * n, coordinates));   // [T, (N−1)²]
        // E = Q Z Qᵀ as a linear map of the chart coordinates: E[i·N + j] = Σ_{a,c} Q[i, a]·Q[j, c]·Z[a·(N−1) + c].
        var q = HelmertBasis(n);
        var map = new float[entries * coordinates];
        var identity = new float[entries];
        var center = new float[entries];
        var negativeInverseCenter = new float[entries];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                int e = i * n + j;
                for (int a = 0; a < nm; a++)
                {
                    for (int c = 0; c < nm; c++)
                    {
                        map[e * coordinates + a * nm + c] = (float)(q[i, a] * q[j, c]);
                    }
                }

                identity[e] = i == j ? 1f : 0f;
                center[e] = (float)(0.5 * ((i == j ? 1.0 : 0.0) + 1.0 / n));
                negativeInverseCenter[e] = -1f / center[e];
            }
        }

        var chart = b.Reshape(b.Emit1("ReduceSum", [b.Mul(b.Reshape(z, 0, 1, coordinates), F32(b, "hc_chart_map", map, new Shape(1, entries, coordinates)))]), 0, entries); // [T, N²]
        var rho = b.Reshape(b.Emit1("ReduceMax", [b.Mul(chart, F32(b, "hc_chart_neg_inv_center", negativeInverseCenter, new Shape(1, entries)))]), 0, 1);
        rho = b.Binary("Max", rho, F32(b, "hc_chart_zero", [0f], new Shape(1, 1)));
        var phi = b.Add(b.Binary("Div", chart, b.Scale(rho, 1f, 1f)), F32(b, "hc_chart_center", center, new Shape(1, entries)));
        var eye = F32(b, "hc_chart_identity", identity, new Shape(1, entries));
        var mixing = b.Add(b.Mul(b.Sub(phi, eye), b.Sigmoid(b.Slice(scale, 0, 3, 1))), eye);
        return b.Reshape(mixing, 0, n, n);
    }

    // The Helmert basis [N, N − 1]: column k − 1 is (1, …, 1, −k, 0, …) / sqrt(k (k + 1)) with k ones.
    private static double[,] HelmertBasis(int n)
    {
        var q = new double[n, n - 1];
        for (int k = 1; k < n; k++)
        {
            double norm = Math.Sqrt(k * (k + 1.0));
            for (int i = 0; i < k; i++)
            {
                q[i, k - 1] = 1.0 / norm;
            }

            q[k, k - 1] = -k / norm;
        }

        return q;
    }

    private static Value F32(GraphBuilder b, string name, float[] values, Shape shape)
        => b.Constant($"{name}_{values.Length}", DataType.F32, shape, System.Runtime.InteropServices.MemoryMarshal.AsBytes(values.AsSpan()).ToArray());

    private static Value NormalizeColumns(GraphBuilder b, Value c, int n, float eps)
    {
        var columnSums = b.Reshape(b.Emit1("ReduceSum", [b.Permute(c, 0, 2, 1)]), 0, 1, n);                             // [T, 1, N]
        return b.Binary("Div", c, b.Scale(columnSums, 1f, eps));
    }
}
