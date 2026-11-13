namespace Sharplight.Llm.Embeddings;

/// <summary>Scores of the documents in input order: <see cref="Scores"/> after the model's activation (sigmoid → [0, 1]), the raw <see cref="Logits"/>.</summary>
public sealed record RerankResult(IReadOnlyList<float> Scores, IReadOnlyList<float> Logits, IReadOnlyList<int> TokenCounts)
{
    /// <summary>Sum of <see cref="TokenCounts"/>.</summary>
    public int TotalTokens => TokenCounts.Sum();
}
