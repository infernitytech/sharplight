using Sharplight.Llm.Embeddings;
using Sharplight.Models;
using Sharplight.Tasks;

namespace Sharplight.Serving.Tests;

/// <summary>Scores a document by the fraction of query words it contains.</summary>
internal sealed class FakeReranker : ITextReranker
{
    public TaskClientMetadata Metadata { get; } = new(ModelKinds.TextReranking, "fake-rerank", "fake");
    public int MaxSequenceLength => 512;

    public Task<RerankResult> RerankAsync(string query, IReadOnlyList<string> documents, RerankOptions? options = null, CancellationToken cancellationToken = default)
        => Task.FromResult(Score(query, documents));

    public object? GetService(Type serviceType, object? serviceKey = null) => null;

    public void Dispose()
    {
    }

    private static RerankResult Score(string query, IReadOnlyList<string> documents)
    {
        var q = query.ToLowerInvariant().Split(' ', StringSplitOptions.RemoveEmptyEntries);
        var scores = documents.Select(d => q.Count(w => d.ToLowerInvariant().Contains(w)) / (float)q.Length).ToList();
        return new RerankResult(scores, scores, [.. documents.Select(d => q.Length + d.Split(' ').Length)]);
    }
}
