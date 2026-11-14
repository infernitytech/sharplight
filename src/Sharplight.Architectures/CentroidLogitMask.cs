using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// A centroid-routed language-model head (Gemma 4 assistants' masked embedder): the vocabulary is split into clusters of equal size, each with a
/// centroid; a row keeps the logits of the tokens in its <c>k</c> best-scoring clusters (scores = hidden · centroid) and masks every other token.
/// The kept logits are the full head's; the masked ones get <see cref="MaskedLogit"/> (the reference fills them with the smallest kept logit minus
/// one, so the argmax and the kept values agree, the masked values do not).
/// </summary>
internal static class CentroidLogitMask
{
    /// <summary>The logit of a masked token.</summary>
    public const float MaskedLogit = -1e30f;

    /// <summary>Masks <paramref name="logits"/> [R, V] to the tokens of each row's best clusters.</summary>
    /// <param name="b">Graph under construction.</param>
    /// <param name="logits">The full head's logits [R, V].</param>
    /// <param name="hidden">The rows the logits were computed from [R, H].</param>
    /// <param name="centroids">Cluster centroids [C, H].</param>
    /// <param name="centroidOfToken">The cluster of every token, i32 [V].</param>
    /// <param name="topK">Clusters kept per row.</param>
    public static Value Apply(GraphBuilder b, Value logits, Value hidden, Value centroids, Value centroidOfToken, int topK)
    {
        var scores = b.MatMul(hidden, centroids, "centroid_scores");                                                      // [R, C]
        var best = b.Emit("TopK", [scores], Attributes.From(("k", (long)topK)), "centroid_topk")[0];                        // [R, k], sorted
        var threshold = b.Slice(best, 1, topK - 1, 1);                                                                     // [R, 1]
        var tokenScores = b.Emit1("GatherAxis", [scores, centroidOfToken], Attributes.From(("axis", 1L)), "token_centroid_scores");  // [R, V]
        var keep = b.Binary("GreaterEqual", tokenScores, threshold);
        var masked = b.Constant("centroid_masked_logit", DataType.F32, new Shape(1), BitConverter.GetBytes(MaskedLogit));
        return b.Emit1("Where", [keep, logits, masked], null, "logits_masked");
    }
}
