using System.Collections.Concurrent;

namespace Sharplight.Runtime;

/// <summary>
/// Key/blob store for compilation artifacts of one device target (kernel binaries, driver pipeline caches).
/// Keys are content hashes (kernel source + macros + target) or provider-defined names, so a hit is always valid for
/// the key; a cache built for another device simply misses. Lets compiled models be saved and reloaded without
/// recompiling (see <see cref="CompileOptions.CacheFor"/>).
/// </summary>
public interface ICompilationCache
{
    bool TryGet(string key, out ReadOnlyMemory<byte> data);
    void Put(string key, ReadOnlyMemory<byte> data);
}

/// <summary>Stored compilation caches, one per device target (<see cref="IComputeDevice.CompilationTarget"/>).</summary>
public interface ICompilationCacheStore
{
    IEnumerable<string> Targets { get; }
    /// <summary>The read-only cache stored for <paramref name="target"/>, or null.</summary>
    ICompilationCache? Find(string target);
}

/// <summary>
/// In-memory cache that records every entry used during compilation (hits from <see cref="Fallback"/> and new
/// entries alike), so the complete set a model needs can be exported afterwards.
/// </summary>
public sealed class CompilationCache(string target, ICompilationCache? fallback = null) : ICompilationCache
{
    readonly ConcurrentDictionary<string, ReadOnlyMemory<byte>> _entries = new(StringComparer.Ordinal);

    public string Target => target;
    public ICompilationCache? Fallback => fallback;
    public IReadOnlyDictionary<string, ReadOnlyMemory<byte>> Entries => _entries;

    public bool TryGet(string key, out ReadOnlyMemory<byte> data)
    {
        if (_entries.TryGetValue(key, out data)) return true;
        if (fallback is not null && fallback.TryGet(key, out data))
        {
            _entries[key] = data;
            return true;
        }
        return false;
    }

    public void Put(string key, ReadOnlyMemory<byte> data) => _entries[key] = data;
}
