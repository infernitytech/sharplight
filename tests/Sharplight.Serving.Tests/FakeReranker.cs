using Sharplight.Llm.Embeddings;

namespace Sharplight.Serving.Tests;

/// <summary>Scores a document by the fraction of query words it contains.</summary>
internal sealed class FakeReranker : ITextReranker
{
    public string Name => "fake-rerank";
    public int MaxSequenceLength => 512;

    public RerankResult Score(string query, IReadOnlyList<string> documents, RerankOptions? options = null)
    {
        var q = query.ToLowerInvariant().Split(' ', StringSplitOptions.RemoveEmptyEntries);
        var scores = documents.Select(d => q.Count(w => d.ToLowerInvariant().Contains(w)) / (float)q.Length).ToList();
        return new RerankResult(scores, scores, [.. documents.Select(d => q.Length + d.Split(' ').Length)]);
    }
}
