using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// Accumulates the canonical weights of one import: rejects two checkpoint tensors mapping to the same canonical name and
/// reports tensors no rule handled in one error instead of failing on the first.
/// </summary>
internal sealed class HfWeightCollector
{
    private const int MaxUnmappedNamesReported = 10;

    private readonly Dictionary<string, WeightInfo> _weights = new(StringComparer.Ordinal);
    private readonly List<string> _unmapped = [];

    /// <summary>The canonical weights added so far.</summary>
    public IReadOnlyDictionary<string, WeightInfo> Weights => _weights;

    /// <summary>Adds a weight under its own name.</summary>
    /// <exception cref="InvalidDataException">The name is already taken.</exception>
    public void Add(WeightInfo weight)
    {
        if (!_weights.TryAdd(weight.Name, weight))
        {
            throw new InvalidDataException($"Two HF tensors map to '{weight.Name}'.");
        }
    }

    /// <summary>Records a checkpoint tensor that has no canonical mapping.</summary>
    public void AddUnmapped(string tensorName) => _unmapped.Add(tensorName);

    /// <summary>Throws when any tensor was recorded as unmapped; <paramref name="label"/> prefixes the message (the model type).</summary>
    /// <exception cref="NotSupportedException">At least one tensor is unmapped.</exception>
    public void ThrowIfUnmapped(string? label)
    {
        if (_unmapped.Count == 0)
        {
            return;
        }

        string shown = string.Join(", ", _unmapped.Take(MaxUnmappedNamesReported));
        string more = _unmapped.Count > MaxUnmappedNamesReported ? ", ..." : "";
        throw new NotSupportedException($"{label}: {_unmapped.Count} tensor(s) have no canonical mapping: {shown}{more}");
    }
}
