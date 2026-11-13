using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// Collects per-expert 2-D tensors (<c>experts.{j}.up_proj.weight</c>, ...) of a mixture-of-experts checkpoint and stacks them into
/// one rank-3 canonical weight [E, N, K] per projection once all tensors have been seen. Stacking is lazy (no copy up front).
/// </summary>
internal sealed class HfExpertStacker
{
    private readonly SortedDictionary<string, SortedDictionary<int, WeightInfo>> _parts = new(StringComparer.Ordinal);

    /// <summary>Records expert <paramref name="expert"/> of the stacked weight <paramref name="canonicalName"/>.</summary>
    public void Add(string canonicalName, int expert, WeightInfo weight)
    {
        if (!_parts.TryGetValue(canonicalName, out var byIndex))
        {
            _parts[canonicalName] = byIndex = [];
        }

        byIndex[expert] = weight;
    }

    /// <summary>Adds every stacked weight to <paramref name="collector"/>.</summary>
    /// <exception cref="InvalidDataException">A projection does not have exactly experts 0 .. <paramref name="expertCount"/>-1.</exception>
    public void StackInto(HfWeightCollector collector, int expertCount)
    {
        foreach (var (name, byIndex) in _parts)
        {
            if (byIndex.Count != expertCount || byIndex.Keys.Max() != expertCount - 1)
            {
                throw new InvalidDataException($"{name}: expected {expertCount} experts, found {byIndex.Count}.");
            }

            collector.Add(WeightInfo.Stack(name, [.. byIndex.Values]));
        }
    }
}
