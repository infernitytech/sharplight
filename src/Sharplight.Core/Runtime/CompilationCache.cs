using System.Collections.Concurrent;

namespace Sharplight.Runtime;

/// <summary>
/// In-memory cache that records every entry used during compilation (hits from <see cref="Fallback"/> and new
/// entries alike), so the complete set a model needs can be exported afterwards.
/// </summary>
public sealed class CompilationCache(string target, ICompilationCache? fallback = null) : ICompilationCache
{
    private readonly ConcurrentDictionary<string, ReadOnlyMemory<byte>> _entries = new(StringComparer.Ordinal);

    /// <summary>The device target this cache belongs to.</summary>
    public string Target => target;
    /// <summary>Cache consulted on a miss (for example one loaded from a saved model), or <see langword="null"/>.</summary>
    public ICompilationCache? Fallback => fallback;
    /// <summary>Everything stored or served so far.</summary>
    public IReadOnlyDictionary<string, ReadOnlyMemory<byte>> Entries => _entries;

    /// <summary>Looks the key up here first, then in <see cref="Fallback"/> (recording a hit).</summary>
    public bool TryGet(string key, out ReadOnlyMemory<byte> data)
    {
        if (_entries.TryGetValue(key, out data))
        {
            return true;
        }

        if (fallback is not null && fallback.TryGet(key, out data))
        {
            _entries[key] = data;
            return true;
        }
        return false;
    }

    /// <summary>Records an entry.</summary>
    public void Put(string key, ReadOnlyMemory<byte> data) => _entries[key] = data;
}
