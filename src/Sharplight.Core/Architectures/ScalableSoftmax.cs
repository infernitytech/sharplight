using Sharplight.Graph;

namespace Sharplight.Architectures;

/// <summary>
/// Scalable softmax (SSMax, Mistral Large 4): every query head is multiplied by <c>s_h · ln(pos + 1) + b_h</c> with learned per-head
/// <c>{prefix}.weight</c> (s, [H]) and <c>{prefix}.bias</c> (b, [H]), which sharpens the attention of late positions. The factor scales the scores of
/// the head linearly, so it applies to any query that is linear in the original one (absorbed latent queries, indexer queries before their relu).
/// </summary>
internal static class ScalableSoftmax
{
    /// <summary>ln(pos + 1) of every row, shaped [T, 1, 1]; build once per graph and share between layers.</summary>
    /// <param name="b">Graph under construction.</param>
    /// <param name="positions">Positions [T, 1] (i32).</param>
    public static Value LogPositions(GraphBuilder b, Value positions)
        => b.Reshape(b.Unary("Log", b.Scale(b.Emit1("Cast", [positions], Attributes.From(("type", "f32"))), 1f, 1f)), 0, 1, 1);

    /// <summary>Multiplies <paramref name="query"/> [T, H, D] by the factors below <paramref name="prefix"/>; unchanged when the weights are absent.</summary>
    /// <param name="b">Graph under construction.</param>
    /// <param name="query">Queries [T, H, D].</param>
    /// <param name="logPositions">The rows' <see cref="LogPositions"/>, or null when the model has no SSMax.</param>
    /// <param name="prefix">Weight name prefix of the factors (e.g. <c>layers.3.attention.ssmax</c>).</param>
    public static Value Apply(GraphBuilder b, Value query, Value? logPositions, string prefix)
    {
        if (logPositions is null || b.TryWeight(prefix + ".weight") is not { } scale)
        {
            return query;
        }

        long heads = scale.Shape[0].Fixed;
        var factors = b.Add(b.Mul(logPositions, b.Reshape(b.Weight(prefix + ".weight"), heads, 1)), b.Reshape(b.Weight(prefix + ".bias"), heads, 1));
        return b.Mul(query, factors, $"{prefix}_scaled");
    }
}
