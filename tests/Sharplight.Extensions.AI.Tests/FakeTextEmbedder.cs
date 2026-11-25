using Sharplight.Llm.Embeddings;
using Sharplight.Models;
using Sharplight.Tasks;

namespace Sharplight.Extensions.AI.Tests;

/// <summary>Embedder whose vector is (text length, 1, 2, 3), truncated to the requested dimensions; records the last options.</summary>
internal sealed class FakeTextEmbedder : ITextEmbedder
{
    public TaskClientMetadata Metadata { get; } = new(ModelKinds.TextEmbedding, "fake", "fake");
    public int Dimensions => 4;
    public int MaxSequenceLength => 100;
    public IReadOnlyDictionary<string, string> Prompts { get; } = new Dictionary<string, string> { ["query"] = "q: " };
    public TextEmbeddingOptions? Last { get; private set; }
    public bool Disposed { get; private set; }

    public int[] Tokenize(string text, TextEmbeddingOptions? options = null) => [.. text.Select(c => (int)c)];

    public Task<EmbeddingResult> EmbedAsync(IReadOnlyList<string> texts, TextEmbeddingOptions? options = null, CancellationToken cancellationToken = default)
    {
        Last = options;
        return Task.FromResult(new EmbeddingResult([.. texts.Select(t => new float[] { t.Length, 1, 2, 3 }.Take(options?.Dimensions ?? 4).ToArray())], [.. texts.Select(t => t.Length)]));
    }

    public Task<EmbeddingResult> EmbedTokensAsync(IReadOnlyList<IReadOnlyList<int>> tokens, TextEmbeddingOptions? options = null, CancellationToken cancellationToken = default)
        => throw new NotSupportedException();

    public object? GetService(Type serviceType, object? serviceKey = null) => null;
    public void Dispose() => Disposed = true;
}
