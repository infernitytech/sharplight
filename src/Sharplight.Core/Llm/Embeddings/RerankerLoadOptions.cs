namespace Sharplight.Llm.Embeddings;

/// <summary>Options for <see cref="Reranker.Load"/>.</summary>
public sealed class RerankerLoadOptions : PackedModelLoadOptions
{
    /// <summary>How pairs are scored (default: from the model).</summary>
    public RerankerKind Kind { get; init; } = RerankerKind.Auto;
    /// <summary>Task instruction of <see cref="RerankerKind.YesNo"/> rerankers (default: the model's, else a web-search instruction).</summary>
    public string? Instruction { get; init; }
}
