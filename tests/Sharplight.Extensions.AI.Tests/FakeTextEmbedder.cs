using Sharplight.Llm.Embeddings;

namespace Sharplight.Extensions.AI.Tests;

/// <summary>Embedder whose vector is (text length, 1, 2, 3), truncated to the requested dimensions; records the last options.</summary>
internal sealed class FakeTextEmbedder : ITextEmbedder, IDisposable
{
    public string Name => "fake";
    public int Dimensions => 4;
    public int MaxSequenceLength => 100;
    public IReadOnlyDictionary<string, string> Prompts { get; } = new Dictionary<string, string> { ["query"] = "q: " };
    public EmbedOptions? Last { get; private set; }
    public bool Disposed { get; private set; }

    public EmbeddingResult Embed(IReadOnlyList<string> texts, EmbedOptions? options = null)
    {
        Last = options;
        return new([.. texts.Select(t => new float[] { t.Length, 1, 2, 3 }.Take(options?.Dimensions ?? 4).ToArray())], [.. texts.Select(t => t.Length)]);
    }

    public EmbeddingResult EmbedTokens(IReadOnlyList<IReadOnlyList<int>> tokens, EmbedOptions? options = null) => throw new NotSupportedException();
    public void Dispose() => Disposed = true;
}
