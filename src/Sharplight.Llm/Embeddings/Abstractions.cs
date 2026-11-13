namespace Sharplight.Llm.Embeddings;

/// <summary>Something that turns texts into embedding vectors (<see cref="EmbeddingModel"/>; fakes in tests, other backends).</summary>
public interface ITextEmbedder
{
    /// <summary>Model name reported to clients.</summary>
    string Name { get; }
    /// <summary>Length of the vectors (before an <see cref="EmbedOptions.Dimensions"/> cut).</summary>
    int Dimensions { get; }
    /// <summary>Longest input in tokens.</summary>
    int MaxSequenceLength { get; }
    /// <summary>Named prompts (e.g. "query", "document") selectable with <see cref="EmbedOptions.Prompt"/>.</summary>
    IReadOnlyDictionary<string, string> Prompts { get; }
    EmbeddingResult Embed(IReadOnlyList<string> texts, EmbedOptions? options = null);
    /// <summary>Embeds pre-tokenized inputs as given.</summary>
    EmbeddingResult EmbedTokens(IReadOnlyList<IReadOnlyList<int>> tokens, EmbedOptions? options = null);
}

/// <summary>Something that scores documents against a query (<see cref="Reranker"/>; fakes in tests, other backends).</summary>
public interface ITextReranker
{
    string Name { get; }
    int MaxSequenceLength { get; }
    RerankResult Score(string query, IReadOnlyList<string> documents, RerankOptions? options = null);
}
