using Sharplight.Llm.Embeddings;

namespace Sharplight.Studio.Tests.Fakes;

/// <summary>An embedder whose vectors count the topic words of a text (one dimension per topic), so similarity follows shared words.</summary>
internal sealed class FakeTextEmbedder : ITextEmbedder
{
    private static readonly string[] s_topics = ["cat", "dog", "ship", "engine", "tax"];

    public string Name => "fake-text-embedder";

    public int Dimensions => s_topics.Length;

    public int MaxSequenceLength => 512;

    public IReadOnlyDictionary<string, string> Prompts { get; } = new Dictionary<string, string> { ["query"] = "query: " };

    /// <summary>The options of every call.</summary>
    public List<EmbedOptions?> Calls { get; } = [];

    /// <summary>Every text embedded so far.</summary>
    public List<string> Texts { get; } = [];

    public EmbeddingResult Embed(IReadOnlyList<string> texts, EmbedOptions? options = null)
    {
        Calls.Add(options);
        Texts.AddRange(texts);
        var vectors = texts.Select(text => s_topics.Select(topic => (float)CountOf(text, topic)).ToArray()).ToList();
        return new EmbeddingResult(vectors, [.. texts.Select(text => text.Length)]);
    }

    public EmbeddingResult EmbedTokens(IReadOnlyList<IReadOnlyList<int>> tokens, EmbedOptions? options = null) => throw new NotSupportedException();

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
