using Sharplight.Graph;

namespace Sharplight.Architectures;

/// <summary>
/// Per-token summaries of a mixture-of-experts routing decision, for models that feed the routing back into later layers: the selected experts'
/// learned identity vectors blended by their mixing weights, and the entropy and the largest probability of the softmax over all router logits.
/// </summary>
internal static class MoeRoutingSignals
{
    // Lower bound of a probability inside the logarithm of the entropy.
    private const float MinProbability = 1e-9f;

    /// <summary>
    /// Σ_k weights[k] · embedding[ids[k]] for the expert embedding <paramref name="embedding"/> [E, R], weights [T, k] and indices [T, k]; returns [T, R].
    /// </summary>
    public static Value Summary(GraphBuilder b, Value embedding, Value weights, Value ids, int k)
    {
        var rows = b.Emit1("GatherAxis", [embedding, ids], Attributes.From(("axis", 0L)));                                      // [T, k, R]
        var weighted = b.Mul(rows, b.Reshape(weights, 0, k, 1));
        long width = embedding.Shape[1].Fixed;
        return b.Reshape(b.Emit1("ReduceSum", [b.Permute(weighted, 0, 2, 1)]), 0, width);
    }

    /// <summary>The entropy −Σ p·log(max(p, 1e-9)) and the largest probability of softmax(<paramref name="logits"/> [T, E]), each [T, 1].</summary>
    public static (Value Entropy, Value Confidence) EntropyAndConfidence(GraphBuilder b, Value logits)
    {
        var probs = b.Emit1("Softmax", [logits]);
        var logs = b.Unary("Log", b.Emit1("Clip", [probs], Attributes.From(("min", MinProbability), ("max", 1f))));
        var entropy = b.Scale(b.Unary("ReduceSum", b.Mul(probs, logs)), -1f);
        return (entropy, b.Unary("ReduceMax", probs));
    }
}
