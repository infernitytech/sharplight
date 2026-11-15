namespace Sharplight.Llm.Embeddings;

/// <summary>How a reranker turns a (query, document) pair into a score.</summary>
public enum RerankerKind
{
    /// <summary>Pick from the model: a classifier head → <see cref="CrossEncoder"/>, otherwise a decoder → <see cref="YesNo"/>.</summary>
    Auto,
    /// <summary>Encoder with a classification head over "query + document" (bge-reranker, ms-marco cross-encoders).</summary>
    CrossEncoder,
    /// <summary>
    /// Decoder judging relevance in a chat prompt; the score is P("yes") from the "yes" / "no" logits of the last token
    /// (Qwen3-Reranker).
    /// </summary>
    YesNo,
}
