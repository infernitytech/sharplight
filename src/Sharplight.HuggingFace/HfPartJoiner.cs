using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// Collects the numbered parts of tensors that a checkpoint splits (per-expert 2-D tensors <c>experts.{j}.up_proj.weight</c>,
/// row shards <c>shard_{j}.weight</c>, ...) and joins them into one canonical weight each once all parts have been seen.
/// Joining is lazy (no copy up front).
/// </summary>
/// <param name="join">Builds the canonical weight from its name and the parts in index order (for example <see cref="WeightInfo.Stack"/>).</param>
/// <param name="partLabel">What a part is, for error messages (for example "experts").</param>
internal sealed class HfPartJoiner(Func<string, IReadOnlyList<WeightInfo>, WeightInfo> join, string partLabel)
{
    private readonly SortedDictionary<string, SortedDictionary<int, WeightInfo>> _parts = new(StringComparer.Ordinal);

    /// <summary>Per-expert tensors stacked into rank-3 weights [E, N, K].</summary>
    public static HfPartJoiner Experts() => new(WeightInfo.Stack, "experts");

    /// <summary>Row shards concatenated along the first axis.</summary>
    public static HfPartJoiner RowShards() => new(WeightInfo.ConcatRows, "shards");

    /// <summary>Records part <paramref name="index"/> of the joined weight <paramref name="canonicalName"/>.</summary>
    public void Add(string canonicalName, int index, WeightInfo weight)
    {
        if (!_parts.TryGetValue(canonicalName, out var byIndex))
        {
            _parts[canonicalName] = byIndex = [];
        }

        byIndex[index] = weight;
    }

    /// <summary>Adds every joined weight to <paramref name="collector"/>.</summary>
    /// <exception cref="InvalidDataException">A weight does not have exactly parts 0 .. <paramref name="partCount"/>-1.</exception>
    public void JoinInto(HfWeightCollector collector, int partCount)
    {
        foreach (var (name, byIndex) in _parts)
        {
            if (byIndex.Count != partCount || byIndex.Keys.Max() != partCount - 1)
            {
                throw new InvalidDataException($"{name}: expected {partCount} {partLabel}, found {byIndex.Count}.");
            }

            collector.Add(join(name, [.. byIndex.Values]));
        }
    }
}
