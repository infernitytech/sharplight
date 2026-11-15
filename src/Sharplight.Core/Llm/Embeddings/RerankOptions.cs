namespace Sharplight.Llm.Embeddings;

/// <summary>Per-call options of <see cref="Reranker.Score"/>.</summary>
public sealed record RerankOptions
{
    /// <summary>Task instruction for <see cref="RerankerKind.YesNo"/> rerankers; null = the load option / model default.</summary>
    public string? Instruction { get; init; }
    /// <summary>Cut pairs longer than <see cref="Reranker.MaxSequenceLength"/> (default); false = reject them.</summary>
    public bool Truncate { get; init; } = true;
}
