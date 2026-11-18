namespace Sharplight.Studio.ViewModels.Search;

/// <summary>The embedded passages of a folder, kept in memory for the model that made them.</summary>
/// <param name="Folder">The indexed folder.</param>
/// <param name="ModelId">The embedding model (vectors of different models are not comparable).</param>
/// <param name="FileCount">How many documents were read.</param>
/// <param name="Chunks">The passages.</param>
internal sealed record SearchIndex(string Folder, string ModelId, int FileCount, IReadOnlyList<IndexedChunk> Chunks)
{
    /// <summary>The <paramref name="count"/> passages closest to <paramref name="query"/>, best first.</summary>
    /// <param name="query">The query's embedding.</param>
    /// <param name="count">How many passages.</param>
    public IReadOnlyList<(IndexedChunk Chunk, float Score)> Nearest(float[] query, int count)
    {
        var unit = Normalize(query);
        return [.. Chunks.Select(chunk => (Chunk: chunk, Score: Dot(unit, chunk.Vector))).OrderByDescending(hit => hit.Score).Take(count)];
    }

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
