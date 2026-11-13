namespace Sharplight.Runtime;

/// <summary>
/// Key/blob store for compilation artifacts of one device target (kernel binaries, driver pipeline caches).
/// Keys are content hashes (kernel source + macros + target) or provider-defined names, so a hit is always valid for
/// the key; a cache built for another device simply misses. Lets compiled models be saved and reloaded without
/// recompiling (see <see cref="CompileOptions.CacheFor"/>).
/// </summary>
public interface ICompilationCache
{
    public bool TryGet(string key, out ReadOnlyMemory<byte> data);
    public void Put(string key, ReadOnlyMemory<byte> data);
}
