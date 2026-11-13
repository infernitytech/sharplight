namespace Sharplight.Runtime;

/// <summary>
/// The cache one device compiles with: routes every key to the artifact set of the tier that claims it
/// (<see cref="ArtifactTarget.Claims"/>), so a device sees one cache while its code and binaries are recorded and stored separately.
/// </summary>
internal sealed class TieredCompilationCache : ICompilationCache
{
    private readonly IReadOnlyList<(ArtifactTarget Target, CompilationCache Set)> _tiers;
    private readonly CompilationCache _unclaimed;

    /// <param name="tiers">The device's targets with their recording sets, in the device's order; not empty.</param>
    public TieredCompilationCache(IReadOnlyList<(ArtifactTarget Target, CompilationCache Set)> tiers)
    {
        _tiers = tiers;
        // Keys no prefix claims go to the catch-all tier, else to the last (most device-specific) one.
        _unclaimed = tiers.FirstOrDefault(t => t.Target.KeyPrefixes.Count == 0).Set ?? tiers[^1].Set;
    }

    public bool TryGet(string key, out ReadOnlyMemory<byte> data) => Route(key).TryGet(key, out data);

    public void Put(string key, ReadOnlyMemory<byte> data) => Route(key).Put(key, data);

    private CompilationCache Route(string key)
    {
        foreach (var (target, set) in _tiers)
        {
            if (target.Claims(key))
            {
                return set;
            }
        }

        return _unclaimed;
    }
}
