using Sharplight.Llm.Embeddings;

namespace Sharplight.Studio.Tests.Fakes;

/// <summary>A reranker that scores a document by the number of words it shares with the query (shared words over document words).</summary>
internal sealed class FakeTextReranker : ITextReranker
{
    public string Name => "fake-text-reranker";

    public int MaxSequenceLength => 512;

    /// <summary>The documents of every call.</summary>
    public List<IReadOnlyList<string>> Calls { get; } = [];

    public RerankResult Score(string query, IReadOnlyList<string> documents, RerankOptions? options = null)
    {
        Calls.Add(documents);
        var words = query.Split(' ', StringSplitOptions.RemoveEmptyEntries).Select(word => word.Trim('?', '.', ',').ToLowerInvariant()).ToHashSet();
        var scores = documents.Select(document =>
        {
            var tokens = document.Split(' ', StringSplitOptions.RemoveEmptyEntries).Select(word => word.Trim('?', '.', ',').ToLowerInvariant()).ToList();
            return tokens.Count == 0 ? 0f : (float)tokens.Count(words.Contains) / tokens.Count;
        }).ToList();
        return new RerankResult(scores, scores, [.. documents.Select(document => document.Length)]);
    }
}
