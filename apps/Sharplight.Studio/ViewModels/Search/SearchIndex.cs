using Sharplight.Studio.Services.Search;

namespace Sharplight.Studio.ViewModels.Search;

/// <summary>The embedded passages of a folder for the model that made them, with the fingerprints of the files they came from.</summary>
/// <param name="Folder">The indexed folder.</param>
/// <param name="ModelId">The embedding model (vectors of different models are not comparable).</param>
/// <param name="FileCount">How many documents were read.</param>
/// <param name="Chunks">The passages.</param>
internal sealed record SearchIndex(string Folder, string ModelId, int FileCount, IReadOnlyList<IndexedChunk> Chunks)
{
    private readonly Lazy<Bm25Index> _keywords = new(() => new Bm25Index([.. Chunks.Select(chunk => chunk.Text)]));

    /// <summary>The documents with their fingerprints, including those that could not be read.</summary>
    public IReadOnlyList<IndexedFile> Files { get; init; } = [];

    /// <summary>The passage length the index was cut with (a different setting means the documents are cut again).</summary>
    public int ChunkCharacters { get; init; }

    /// <summary>The passage overlap the index was cut with.</summary>
    public int OverlapCharacters { get; init; }

    /// <summary>The <paramref name="count"/> passages closest to <paramref name="query"/>, best first.</summary>
    /// <param name="query">The query's embedding.</param>
    /// <param name="count">How many passages.</param>
    public IReadOnlyList<(IndexedChunk Chunk, float Score)> Nearest(float[] query, int count) => [.. Rank(query, count).Select(hit => (Chunks[hit.Index], hit.Score))];

    /// <summary>Positions in <see cref="Chunks"/> of the passages closest to <paramref name="query"/>, best first, with their cosine similarity.</summary>
    /// <param name="query">The query's embedding.</param>
    /// <param name="count">How many passages.</param>
    public IReadOnlyList<(int Index, float Score)> Rank(float[] query, int count)
    {
        var unit = Normalize(query);
        return [.. Enumerable.Range(0, Chunks.Count).Select(i => (Index: i, Score: Dot(unit, Chunks[i].Vector))).OrderByDescending(hit => hit.Score).Take(count)];
    }

    /// <summary>Positions in <see cref="Chunks"/> of the passages that best match the words of <paramref name="query"/> (BM25), best first.</summary>
    /// <param name="query">The question.</param>
    /// <param name="count">How many passages.</param>
    public IReadOnlyList<(int Index, double Score)> Keyword(string query, int count) => _keywords.Value.Search(query, count);

    /// <summary>A copy of <paramref name="vector"/> of length 1 (the zero vector stays zero).</summary>
    /// <param name="vector">An embedding.</param>
    public static float[] Normalize(float[] vector)
    {
        double sum = 0;
        foreach (float value in vector)
        {
            sum += (double)value * value;
        }

        float scale = sum > 0 ? (float)(1 / Math.Sqrt(sum)) : 0;
        return [.. vector.Select(value => value * scale)];
    }

    private static float Dot(float[] a, float[] b)
    {
        float sum = 0;
        int length = Math.Min(a.Length, b.Length);
        for (int i = 0; i < length; i++)
        {
            sum += a[i] * b[i];
        }

        return sum;
    }
}
