namespace Sharplight.Llm.Embeddings;

/// <summary>Something that scores documents against a query (<see cref="Reranker"/>; fakes in tests, other backends).</summary>
public interface ITextReranker
{
    /// <summary>Model name reported to clients.</summary>
    public string Name { get; }
    /// <summary>Longest (query, document) input in tokens.</summary>
    public int MaxSequenceLength { get; }
    /// <summary>Scores every document against the query; results are in document order.</summary>
    public RerankResult Score(string query, IReadOnlyList<string> documents, RerankOptions? options = null);
}
