namespace Sharplight.Llm.Embeddings;

/// <summary>Embeddings of a batch, in input order, plus token counts.</summary>
public sealed record EmbeddingResult(IReadOnlyList<float[]> Vectors, IReadOnlyList<int> TokenCounts)
{
    /// <summary>Sum of <see cref="TokenCounts"/>.</summary>
    public int TotalTokens => TokenCounts.Sum();
}
