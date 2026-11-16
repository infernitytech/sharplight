using Sharplight.Graph;

namespace Sharplight.Architectures;

/// <summary>
/// Static hyper-connections over S parallel residual streams held as [T, S, H]: before a sublayer the streams are mixed by a learned S×S matrix and
/// a learned combination of them is normalized into the sublayer input; after it every mixed stream receives the sublayer output scaled by its own
/// learned weight. The coefficients are per pass of a multi-pass stack (they do not depend on the token, unlike <see cref="StreamHyperConnections"/>).
/// </summary>
/// <remarks>
/// Weights below a sublayer prefix <c>p</c> (e.g. <c>layers.3.streams.attention.</c>): <c>p + "mix"</c> [P, S, S] (row i of pass p: the
/// weights of the streams in mixed stream i), <c>p + "read"</c> [P, S] (the sublayer input combination) and <c>p + "write"</c> [P, S] (the
/// sublayer output's weight in each stream).
/// </remarks>
internal static class StaticStreamMixing
{
    /// <summary>Mixes the streams <paramref name="x"/> [T, S, H] for pass <paramref name="pass"/> and returns them with the RMS-normalized sublayer input [T, H].</summary>
    public static (Value Mixed, Value Input) Read(GraphBuilder b, Value x, string p, int pass, int streams, int hidden, Value normWeight, float eps)
    {
        var mix = b.Slice(b.Weight(p + "mix"), 0, pass, 1);                                                                    // [1, S, S]
        var mixed = b.Emit1("BatchMatMul", [mix, x], null, $"{p}mixed_{pass}");                                                  // [T, S, H]
        var read = b.Reshape(b.Slice(b.Weight(p + "read"), 0, pass, 1), 1, 1, streams);                                         // [1, 1, S]
        var input = b.Reshape(b.Emit1("BatchMatMul", [read, x]), 0, hidden);                                                    // [T, H]
        return (mixed, b.RmsNorm(input, normWeight, eps, $"{p}input_{pass}"));
    }

    /// <summary>Adds the sublayer output <paramref name="update"/> [T, H], scaled per stream, to the mixed streams [T, S, H].</summary>
    public static Value Write(GraphBuilder b, Value mixed, string p, int pass, int streams, int hidden, Value update)
    {
        var write = b.Reshape(b.Slice(b.Weight(p + "write"), 0, pass, 1), streams, 1);                                          // [S, 1]
        return b.Add(mixed, b.Mul(b.Reshape(update, 0, 1, hidden), write), $"{p}written_{pass}");
    }

    /// <summary>The mean of the streams [T, S, H] → [T, H].</summary>
    public static Value Mean(GraphBuilder b, Value x, int streams, int hidden)
        => b.Reshape(b.Scale(b.Emit1("ReduceSum", [b.Permute(x, 0, 2, 1)]), 1f / streams), 0, hidden);
}
