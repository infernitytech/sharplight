namespace Sharplight.Llm.Embeddings;

/// <summary>Something that turns texts into embedding vectors (<see cref="EmbeddingModel"/>; fakes in tests, other backends).</summary>
public interface ITextEmbedder
{
    /// <summary>Model name reported to clients.</summary>
    public string Name { get; }
    /// <summary>Length of the vectors (before an <see cref="TextEmbeddingOptions.Dimensions"/> cut).</summary>
    public int Dimensions { get; }
    /// <summary>Longest input in tokens.</summary>
    public int MaxSequenceLength { get; }
    /// <summary>Named prompts (e.g. "query", "document") selectable with <see cref="TextEmbeddingOptions.Prompt"/>.</summary>
    public IReadOnlyDictionary<string, string> Prompts { get; }
    /// <summary>Embeds texts (batched); vectors are in input order.</summary>
    public EmbeddingResult Embed(IReadOnlyList<string> texts, TextEmbeddingOptions? options = null);
    /// <summary>Embeds pre-tokenized inputs as given.</summary>
    public EmbeddingResult EmbedTokens(IReadOnlyList<IReadOnlyList<int>> tokens, TextEmbeddingOptions? options = null);
}
