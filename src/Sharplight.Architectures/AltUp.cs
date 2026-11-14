using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Alternating updates (AltUp, Gemma 3n): the residual is S streams [T, S·H] (stream-major), of which only the active stream 0 runs through
/// each layer. Before a layer the streams are mixed into predictions with per-token coefficients, after it every prediction is corrected
/// by the layer's innovation (its output minus the active prediction). Pure graph code over standard ops, usable by any adapter.
/// </summary>
/// <remarks>
/// Weights: <c>altup.projections.{s}.weight</c> and <c>altup.unembed_projections.{s}.weight</c> [H, H] for s = 0 … S−2 (streams 1 … S−1);
/// per layer below a prefix <c>altup.router_norm.weight</c> [H], <c>altup.router.weight</c> [S, H], <c>altup.prediction_coefs.weight</c> [S², S],
/// <c>altup.correction_coefs.weight</c> [S, S] and optionally <c>altup.correct_output_scale</c> [H]. With the router modalities
/// m(x) = tanh(router(rms_norm(x)·w / H)) [T, S]:
/// <list type="bullet">
/// <item>predict: P = prediction_coefs(m(X₀)) as [T, S, S] (row i, column j), prediction_i = X_i + Σ_j P[i, j]·X_j;</item>
/// <item>correct: c = correction_coefs(m(y)) + 1 [T, S], corrected_i = prediction_i + c_i·(y − prediction₀) for the layer output y;</item>
/// <item>expand: stream 0 is the embedding e, stream s the projection of e rescaled to e's RMS (each RMS at least √1e-5);</item>
/// <item>collapse: the mean of stream 0 and the unembedding projections of the other streams, each rescaled to stream 0's RMS.</item>
/// </list>
/// </remarks>
internal static class AltUp
{
    private const float MagnitudeEps = 1e-5f;

    /// <summary>Widens the embeddings x [T, H] to the streams [T, S·H].</summary>
    public static Value Expand(GraphBuilder b, Value x, int streams, int hidden)
    {
        var target = Rms(b, x, hidden);
        var parts = new List<Value> { x };
        for (int s = 1; s < streams; s++)
        {
            parts.Add(Rescaled(b, b.Linear(x, $"altup.projections.{s - 1}.weight"), target, hidden));
        }

        return b.Concat(1, [.. parts]);
    }

    /// <summary>Collapses the streams [T, S·H] into the hidden state [T, H] before the output norm.</summary>
    public static Value Collapse(GraphBuilder b, Value x, int streams, int hidden)
    {
        var first = Stream(b, x, 0, hidden);
        var target = Rms(b, first, hidden);
        var sum = first;
        for (int s = 1; s < streams; s++)
        {
            sum = b.Add(sum, Rescaled(b, b.Linear(Stream(b, x, s, hidden), $"altup.unembed_projections.{s - 1}.weight"), target, hidden));
        }

        return b.Scale(sum, 1f / streams, 0, "altup_collapsed");
    }

    /// <summary>The predictions [T, S·H] of the layer whose AltUp weights are below <paramref name="prefix"/>.</summary>
    public static Value Predict(GraphBuilder b, Value x, string prefix, int streams, int hidden, float eps)
    {
        var modalities = Modalities(b, Stream(b, x, 0, hidden), prefix, hidden, eps);
        var coefficients = b.Reshape(b.Linear(modalities, prefix + "altup.prediction_coefs.weight"), 0, streams, streams);   // [T, S, S]
        var streamsOf = b.Reshape(x, 0, streams, hidden);
        var mixed = b.Emit1("BatchMatMul", [coefficients, streamsOf], null, $"{prefix}altup_mixed");                         // [T, S, H]
        return b.Reshape(b.Add(streamsOf, mixed), 0, (long)streams * hidden);
    }

    /// <summary>The corrected streams [T, S·H] from the predictions and the active stream's layer output y [T, H].</summary>
    public static Value Correct(GraphBuilder b, Value predictions, Value y, string prefix, int streams, int hidden, float eps)
    {
        var modalities = Modalities(b, y, prefix, hidden, eps);
        var coefficients = b.Reshape(b.Scale(b.Linear(modalities, prefix + "altup.correction_coefs.weight"), 1f, 1f), 0, streams, 1);   // [T, S, 1]
        var innovation = b.Reshape(b.Sub(y, Stream(b, predictions, 0, hidden)), 0, 1, hidden);
        var corrected = b.Add(b.Reshape(predictions, 0, streams, hidden), b.Mul(coefficients, innovation), $"{prefix}altup_corrected");
        return b.Reshape(corrected, 0, (long)streams * hidden);
    }

    /// <summary>Adds <paramref name="delta"/> [T, H] to every stream but the active one.</summary>
    public static Value AddToInactive(GraphBuilder b, Value x, Value delta, int streams, int hidden, string name)
    {
        var mask = new float[streams];
        Array.Fill(mask, 1f, 1, streams - 1);
        var bytes = new byte[streams * 4];
        Buffer.BlockCopy(mask, 0, bytes, 0, bytes.Length);
        var inactive = b.Constant("altup_inactive", DataType.F32, new Shape(1, streams, 1), bytes);
        var sum = b.Add(b.Reshape(x, 0, streams, hidden), b.Mul(inactive, b.Reshape(delta, 0, 1, hidden)), name);
        return b.Reshape(sum, 0, (long)streams * hidden);
    }

    /// <summary>Stream <paramref name="s"/> [T, H] of the streams [T, S·H].</summary>
    public static Value Stream(GraphBuilder b, Value x, int s, int hidden) => b.Slice(x, 1, (long)s * hidden, hidden);

    // tanh(router(rms_norm(x)·w / H)) [T, S].
    private static Value Modalities(GraphBuilder b, Value x, string prefix, int hidden, float eps)
    {
        var normed = b.Scale(b.RmsNorm(x, b.Weight(prefix + "altup.router_norm.weight"), eps), 1f / hidden);
        return b.Unary("Tanh", b.Linear(normed, prefix + "altup.router.weight"));
    }

    // sqrt(mean(x²)) [T, 1].
    private static Value Rms(GraphBuilder b, Value x, int hidden)
        => b.Unary("Sqrt", b.Scale(b.Emit1("ReduceSum", [b.Mul(x, x)]), 1f / hidden));

    // y · target / max(rms(y), √eps).
    private static Value Rescaled(GraphBuilder b, Value y, Value target, int hidden)
    {
        var meanSquare = b.Scale(b.Emit1("ReduceSum", [b.Mul(y, y)]), 1f / hidden);
        var magnitude = b.Unary("Sqrt", b.Binary("Max", meanSquare, b.Constant("altup_eps", DataType.F32, new Shape(1, 1), BitConverter.GetBytes(MagnitudeEps))));
        return b.Mul(y, b.Binary("Div", target, magnitude));
    }
}
