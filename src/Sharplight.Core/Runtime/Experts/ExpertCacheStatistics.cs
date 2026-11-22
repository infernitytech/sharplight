namespace Sharplight.Runtime.Experts;

/// <summary>Counters of an <see cref="ExpertStore"/> since it was created or <see cref="ExpertStore.ResetStatistics"/> was called.</summary>
/// <param name="Hits">Experts a pass found in the cache (filled, or being filled by a prefetch).</param>
/// <param name="Misses">Experts a pass had to read.</param>
/// <param name="Prefetched">Experts read ahead of the pass that needs them (the layer's later products, the next chunk of a large pass).</param>
/// <param name="Evictions">Cached experts dropped for others.</param>
/// <param name="BytesRead">Bytes read from disk.</param>
/// <param name="ReadWait">Time passes waited for reads.</param>
/// <param name="CacheBytes">Bytes of the cache slots.</param>
/// <param name="CacheSlots">Cache slots.</param>
public readonly record struct ExpertCacheStatistics(long Hits, long Misses, long Prefetched, long Evictions, long BytesRead, TimeSpan ReadWait, long CacheBytes, int CacheSlots)
{
    /// <summary>Share of the requested experts that were cached, 0 to 1 (0 before the first request).</summary>
    public double HitRate => Hits + Misses == 0 ? 0 : (double)Hits / (Hits + Misses);

    /// <inheritdoc/>
    public override string ToString()
        => $"hit rate {HitRate:P1} ({Hits} hits, {Misses} misses, {Prefetched} prefetched, {Evictions} evictions), read {BytesRead / 1e9:F2} GB, "
           + $"waited {ReadWait.TotalSeconds:F2} s, cache {CacheBytes / (double)(1L << 30):F1} GiB in {CacheSlots} slots";
}
