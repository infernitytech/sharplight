namespace Sharplight.Runtime;

/// <summary>
/// Key/blob store for compilation artifacts (kernel code, driver pipeline caches).
/// Keys are content hashes (kernel source + macros + target) or provider-defined names, so a hit is always valid for
/// the key; a cache built for another device simply misses. Lets compiled models be saved and reloaded without
/// recompiling (see <see cref="CompileOptions.CacheFor"/>).
/// </summary>
public interface ICompilationCache
{
    /// <summary>
    /// Looks up an artifact.
    /// </summary>
    /// <param name="key">Content hash or provider-defined name.</param>
    /// <param name="data">The stored bytes when found.</param>
    /// <returns>Whether the key was found.</returns>
    public bool TryGet(string key, out ReadOnlyMemory<byte> data);
    /// <summary>Stores an artifact. Implementations must be safe to call from several threads.</summary>
    public void Put(string key, ReadOnlyMemory<byte> data);
}
