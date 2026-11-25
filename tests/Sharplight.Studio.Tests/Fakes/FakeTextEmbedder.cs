using Sharplight.Llm.Embeddings;
using Sharplight.Models;
using Sharplight.Tasks;

namespace Sharplight.Studio.Tests.Fakes;

/// <summary>An embedder whose vectors count the topic words of a text (one dimension per topic), so similarity follows shared words.</summary>
internal sealed class FakeTextEmbedder : ITextEmbedder
{
    private static readonly string[] s_topics = ["cat", "dog", "ship", "engine", "tax"];

    public TaskClientMetadata Metadata { get; } = new(ModelKinds.TextEmbedding, "fake-text-embedder", "fake");

    public int Dimensions => s_topics.Length;

    public int MaxSequenceLength => 512;

    public IReadOnlyDictionary<string, string> Prompts { get; } = new Dictionary<string, string> { ["query"] = "query: " };

    /// <summary>The options of every call.</summary>
    public List<TextEmbeddingOptions?> Calls { get; } = [];

    /// <summary>Every text embedded so far.</summary>
    public List<string> Texts { get; } = [];

    public int[] Tokenize(string text, TextEmbeddingOptions? options = null) => [.. text.Select(c => (int)c)];

    public Task<EmbeddingResult> EmbedAsync(IReadOnlyList<string> texts, TextEmbeddingOptions? options = null, CancellationToken cancellationToken = default)
        => Task.FromResult(Embed(texts, options));

    public Task<EmbeddingResult> EmbedTokensAsync(IReadOnlyList<IReadOnlyList<int>> tokens, TextEmbeddingOptions? options = null, CancellationToken cancellationToken = default)
        => throw new NotSupportedException();

    public object? GetService(Type serviceType, object? serviceKey = null) => null;

    public void Dispose()
    {
    }

    private EmbeddingResult Embed(IReadOnlyList<string> texts, TextEmbeddingOptions? options)
    {
        Calls.Add(options);
        Texts.AddRange(texts);
        var vectors = texts.Select(text => s_topics.Select(topic => (float)CountOf(text, topic)).ToArray()).ToList();
        return new EmbeddingResult(vectors, [.. texts.Select(text => text.Length)]);
    }

    private static int CountOf(string text, string word)
    {
        int count = 0;
        for (int at = text.IndexOf(word, StringComparison.OrdinalIgnoreCase); at >= 0; at = text.IndexOf(word, at + word.Length, StringComparison.OrdinalIgnoreCase))
        {
            count++;
        }

        return count;
    }
}
