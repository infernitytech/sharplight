using Sharplight.Llm.Embeddings;
using Sharplight.Models;
using Sharplight.Tasks;

namespace Sharplight.Serving.Tests;

/// <summary>Deterministic embedder: one token per character, vector = normalized character histogram over <see cref="Dimensions"/> buckets.</summary>
internal sealed class FakeEmbedder : ITextEmbedder
{
    public TaskClientMetadata Metadata { get; } = new(ModelKinds.TextEmbedding, "fake-embed", "fake");
    public int Dimensions => 8;
    public int MaxSequenceLength => 16;
    public IReadOnlyDictionary<string, string> Prompts { get; } = new Dictionary<string, string> { ["query"] = "q: " };
    public List<TextEmbeddingOptions> Calls { get; } = [];

    public int[] Tokenize(string text, TextEmbeddingOptions? options = null)
    {
        string prefix = options?.Prompt is { } p ? Prompts[p] : "";
        return [.. (prefix + text).Select(c => (int)c)];
    }

    public Task<EmbeddingResult> EmbedAsync(IReadOnlyList<string> texts, TextEmbeddingOptions? options = null, CancellationToken cancellationToken = default)
        => EmbedTokensAsync([.. texts.Select(t => (IReadOnlyList<int>)Tokenize(t, options))], options, cancellationToken);

    public Task<EmbeddingResult> EmbedTokensAsync(IReadOnlyList<IReadOnlyList<int>> tokens, TextEmbeddingOptions? options = null, CancellationToken cancellationToken = default)
    {
        try
        {
            return Task.FromResult(EmbedTokens(tokens, options));
        }
        catch (Exception error)
        {
            return Task.FromException<EmbeddingResult>(error);
        }
    }

    public object? GetService(Type serviceType, object? serviceKey = null) => null;

    public void Dispose()
    {
    }

    private EmbeddingResult EmbedTokens(IReadOnlyList<IReadOnlyList<int>> tokens, TextEmbeddingOptions? options)
    {
        options ??= new TextEmbeddingOptions();
        Calls.Add(options);
        var vectors = new List<float[]>();
        foreach (var t in tokens)
        {
            if (t.Count > MaxSequenceLength && !options.Truncate)
            {
                throw new ArgumentException($"Input has {t.Count} tokens; the model accepts at most {MaxSequenceLength}.");
            }

            var v = new float[options.Dimensions ?? Dimensions];
            foreach (int id in t.Take(MaxSequenceLength))
            {
                v[id % v.Length] += 1;
            }

            float n = MathF.Sqrt(v.Sum(x => x * x));
            for (int i = 0; i < v.Length; i++)
            {
                v[i] = n > 0 ? v[i] / n : 0;
            }

            vectors.Add(v);
        }
        return new EmbeddingResult(vectors, [.. tokens.Select(t => Math.Min(t.Count, MaxSequenceLength))]);
    }
}
