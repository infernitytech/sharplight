using System.Collections.Concurrent;

namespace Sharplight.Runtime;

/// <summary>
/// In-memory artifact set for one descriptor that records every entry used during compilation (hits from
/// <see cref="Fallbacks"/> and new entries alike), so the complete set a model needs can be exported afterwards.
/// </summary>
/// <param name="descriptor">Descriptor stored with the set (see <see cref="ArtifactDescriptorKeys"/>).</param>
/// <param name="fallbacks">Caches consulted in order on a miss (for example stored sets of a loaded model), or <see langword="null"/>.</param>
public sealed class CompilationCache(IReadOnlyDictionary<string, string> descriptor, IEnumerable<ICompilationCache>? fallbacks = null) : ICompilationCache
{
    private readonly ConcurrentDictionary<string, ReadOnlyMemory<byte>> _entries = new(StringComparer.Ordinal);
    private readonly ICompilationCache[] _fallbacks = fallbacks is null ? [] : [.. fallbacks];

    /// <summary>Descriptor of the set (what the entries are and which devices may use them).</summary>
    public IReadOnlyDictionary<string, string> Descriptor { get; } = new Dictionary<string, string>(descriptor, StringComparer.Ordinal);
    /// <summary>Caches consulted on a miss, in order.</summary>
    public IReadOnlyList<ICompilationCache> Fallbacks => _fallbacks;
    /// <summary>Everything stored or served so far.</summary>
    public IReadOnlyDictionary<string, ReadOnlyMemory<byte>> Entries => _entries;
    /// <summary>
    /// Number of entries whose bytes came from <see cref="Fallbacks"/> and were not replaced by <see cref="Put"/> since: how much a
    /// stored set saved. Equal to the entry count when the set holds nothing new.
    /// </summary>
    public int FallbackHits => _fromFallbacks.Count;

    private readonly ConcurrentDictionary<string, bool> _fromFallbacks = new(StringComparer.Ordinal);

    /// <summary>Looks the key up here first, then in <see cref="Fallbacks"/> (recording a hit).</summary>
    public bool TryGet(string key, out ReadOnlyMemory<byte> data)
    {
        if (_entries.TryGetValue(key, out data))
        {
            return true;
        }

        foreach (var fallback in _fallbacks)
        {
            if (fallback.TryGet(key, out data))
            {
                if (_entries.TryAdd(key, data))
                {
                    _fromFallbacks[key] = true;
                }

                return true;
            }
        }

        return false;
    }

    /// <summary>Records an entry.</summary>
    public void Put(string key, ReadOnlyMemory<byte> data)
    {
        _entries[key] = data;
        _fromFallbacks.TryRemove(key, out _);
    }
}
