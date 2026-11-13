using System.Collections.Concurrent;

namespace Sharplight.Runtime;

/// <summary>
/// In-memory cache that records every entry used during compilation (hits from <see cref="Fallback"/> and new
/// entries alike), so the complete set a model needs can be exported afterwards.
/// </summary>
public sealed class CompilationCache(string target, ICompilationCache? fallback = null) : ICompilationCache
{
    private readonly ConcurrentDictionary<string, ReadOnlyMemory<byte>> _entries = new(StringComparer.Ordinal);

    public string Target => target;
    public ICompilationCache? Fallback => fallback;
    public IReadOnlyDictionary<string, ReadOnlyMemory<byte>> Entries => _entries;

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

    public void Put(string key, ReadOnlyMemory<byte> data) => _entries[key] = data;
}
