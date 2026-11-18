namespace Sharplight.Studio.Services.Search;

/// <summary>Reciprocal rank fusion: combines several rankings by summing <c>1 / (k + rank)</c> per item.</summary>
internal static class RankFusion
{
    /// <summary>The usual damping constant.</summary>
    public const int DefaultK = 60;

    /// <summary>Fuses <paramref name="rankings"/> (each best first) into one list, best first.</summary>
    /// <param name="rankings">The rankings of item ids.</param>
    /// <param name="k">The damping constant: the larger, the flatter the weights of the ranks.</param>
    /// <returns>Items with their fused score; equal scores keep the item that ranks higher somewhere, then the lower id.</returns>
    public static IReadOnlyList<(int Id, double Score)> Fuse(IReadOnlyList<IReadOnlyList<int>> rankings, int k = DefaultK)
    {
        var scores = new Dictionary<int, double>();
        var best = new Dictionary<int, int>();
        foreach (var ranking in rankings)
        {
            for (int rank = 0; rank < ranking.Count; rank++)
            {
                int id = ranking[rank];
                scores[id] = scores.GetValueOrDefault(id) + 1.0 / (k + rank + 1);
                best[id] = best.TryGetValue(id, out int earlier) ? Math.Min(earlier, rank) : rank;
            }
        }

        return [.. scores.OrderByDescending(pair => pair.Value).ThenBy(pair => best[pair.Key]).ThenBy(pair => pair.Key).Select(pair => (pair.Key, pair.Value))];
    }
}
