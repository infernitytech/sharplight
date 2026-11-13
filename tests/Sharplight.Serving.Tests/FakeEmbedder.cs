using Sharplight.Llm.Embeddings;

namespace Sharplight.Serving.Tests;

/// <summary>Deterministic embedder: one token per character, vector = normalized character histogram over <see cref="Dimensions"/> buckets.</summary>
internal sealed class FakeEmbedder : ITextEmbedder
{
    public string Name => "fake-embed";
    public int Dimensions => 8;
    public int MaxSequenceLength => 16;
    public IReadOnlyDictionary<string, string> Prompts { get; } = new Dictionary<string, string> { ["query"] = "q: " };
    public List<EmbedOptions> Calls { get; } = [];

    public EmbeddingResult Embed(IReadOnlyList<string> texts, EmbedOptions? options = null)
    {
        options ??= new EmbedOptions();
        string prefix = options.Prompt is { }
        p ? Prompts[p] : "";
        return EmbedTokens([.. texts.Select(t => (IReadOnlyList<int>)[.. (prefix + t).Select(c => (int)c)])], options);
    }

    public EmbeddingResult EmbedTokens(IReadOnlyList<IReadOnlyList<int>> tokens, EmbedOptions? options = null)
    {
        options ??= new EmbedOptions();
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
