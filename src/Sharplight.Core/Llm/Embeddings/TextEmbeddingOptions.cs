namespace Sharplight.Llm.Embeddings;

/// <summary>Per-request options of <see cref="ITextEmbedder.EmbedAsync"/> and <see cref="ITextEmbedder.EmbedTokensAsync"/>.</summary>
public sealed record TextEmbeddingOptions
{
    /// <summary>Name of one of the model's prompts (<see cref="ITextEmbedder.Prompts"/>, e.g. "query"); null = the model's default prompt.</summary>
    public string? Prompt { get; init; }
    /// <summary>Literal text prepended to every input (instead of a named prompt), e.g. "query: " for e5 models.</summary>
    public string? Prefix { get; init; }
    /// <summary>Keep only the first n dimensions (Matryoshka models), re-normalized when the model normalizes. Null = all.</summary>
    public int? Dimensions { get; init; }
    /// <summary>Cut inputs longer than <see cref="ITextEmbedder.MaxSequenceLength"/> (default); false = reject them.</summary>
    public bool Truncate { get; init; } = true;
}
