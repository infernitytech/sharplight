using Sharplight.Llm.Embeddings;
using Sharplight.Models;
using Sharplight.Tasks;

namespace Sharplight.Studio.Tests.Fakes;

/// <summary>A reranker that scores a document by the number of words it shares with the query (shared words over document words).</summary>
internal sealed class FakeTextReranker : ITextReranker
{
    public TaskClientMetadata Metadata { get; } = new(ModelKinds.TextReranking, "fake-text-reranker", "fake");

    public int MaxSequenceLength => 512;

    /// <summary>The documents of every call.</summary>
    public List<IReadOnlyList<string>> Calls { get; } = [];

    public Task<RerankResult> RerankAsync(string query, IReadOnlyList<string> documents, RerankOptions? options = null, CancellationToken cancellationToken = default)
        => Task.FromResult(Score(query, documents));

    public object? GetService(Type serviceType, object? serviceKey = null) => null;

    public void Dispose()
    {
    }

    private RerankResult Score(string query, IReadOnlyList<string> documents)
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
