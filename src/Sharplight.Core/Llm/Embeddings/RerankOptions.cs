namespace Sharplight.Llm.Embeddings;

/// <summary>Per-request options of <see cref="ITextReranker.RerankAsync"/>.</summary>
public sealed record RerankOptions
{
    /// <summary>Task instruction of decoder (yes / no) rerankers such as Qwen3-Reranker; null = the model's, else a web-search instruction.</summary>
    public string? Instruction { get; init; }
    /// <summary>Cut pairs longer than <see cref="ITextReranker.MaxSequenceLength"/> (default); false = reject them.</summary>
    public bool Truncate { get; init; } = true;
}
