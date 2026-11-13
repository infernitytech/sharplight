namespace Sharplight.Runtime;

/// <summary>
/// The precompiled artifacts of one model across devices: gives each device a cache (<see cref="For"/>, to use as
/// <see cref="CompileOptions.CacheFor"/>) that is seeded from the matching stored sets and records everything compilation uses,
/// split by tier, so the sets can be exported (<see cref="Export"/>) and stored with the model. Thread-safe.
/// </summary>
/// <param name="stored">Artifact sets stored with the model, or <see langword="null"/>.</param>
public sealed class CompilationArtifacts(ICompilationCacheStore? stored = null)
{
    private readonly Lock _lock = new();
    private readonly Dictionary<IComputeDevice, ICompilationCache?> _devices = new(ReferenceEqualityComparer.Instance);
    /// <summary>Recording sets by target id; devices with equal descriptors (two identical GPUs' code tier) share one.</summary>
    private readonly Dictionary<string, CompilationCache> _sets = new(StringComparer.Ordinal);
    private readonly List<CompilationCache> _order = [];

    /// <summary>The stored sets the caches are seeded from, or <see langword="null"/>.</summary>
    public ICompilationCacheStore? Stored => stored;

    /// <summary>
    /// The cache <paramref name="device"/> compiles with; <see langword="null"/> when the device declares no
    /// <see cref="IComputeDevice.ArtifactTargets"/> (it has nothing to precompile, e.g. the CPU).
    /// </summary>
    public ICompilationCache? For(IComputeDevice device)
    {
        lock (_lock)
        {
            if (_devices.TryGetValue(device, out var cache))
            {
                return cache;
            }

            var targets = device.ArtifactTargets;
            if (targets.Count > 0)
            {
                cache = new TieredCompilationCache([.. targets.Select(t => (t, SetFor(device, t)))]);
            }

            _devices[device] = cache;
            return cache;
        }
    }

    /// <summary>
    /// The sets recorded so far, as they are (without device-level state, see <see cref="Export"/>): e.g. to report how many entries
    /// came from the stored sets (<see cref="CompilationCache.FallbackHits"/>).
    /// </summary>
    public IReadOnlyList<CompilationCache> Recorded
    {
        get
        {
            lock (_lock)
            {
                return [.. _order];
            }
        }
    }

    /// <summary>
    /// The non-empty artifact sets recorded so far, after adding device-level state (driver pipeline caches,
    /// <see cref="IComputeDevice.ExportCompilationState"/>) of every device that compiled with them.
    /// </summary>
    /// <param name="tiers">Only sets of these tiers (<see cref="ArtifactTiers"/>), or <see langword="null"/> for all.</param>
    public IReadOnlyList<CompilationCache> Export(IReadOnlySet<string>? tiers = null)
    {
        lock (_lock)
        {
            foreach (var (device, cache) in _devices)
            {
                if (cache is not null)
                {
                    device.ExportCompilationState(cache);
                }
            }

            return [.. _order.Where(s => s.Entries.Count > 0 && (tiers is null || tiers.Contains(ArtifactDescriptor.Get(s.Descriptor, ArtifactDescriptorKeys.Tier))))];
        }
    }

    /// <summary>
    /// Stored sets <paramref name="device"/> may consult for <paramref name="target"/>: those whose descriptor matches
    /// (<see cref="ArtifactTarget.Matches"/>), plus legacy kernel sets (<see cref="ArtifactTiers.Mixed"/>) stored for the device's
    /// <see cref="IComputeDevice.CompilationTarget"/>. Newest first.
    /// </summary>
    public static IReadOnlyList<StoredArtifactSet> Matching(ICompilationCacheStore store, IComputeDevice device, ArtifactTarget target)
        => [.. store.ArtifactSets.Where(s => target.Matches(s.Descriptor) || IsLegacyFor(s, device, target))];

    private static bool IsLegacyFor(StoredArtifactSet set, IComputeDevice device, ArtifactTarget target)
        => set.Tier == ArtifactTiers.Mixed
            && ArtifactDescriptor.Get(set.Descriptor, ArtifactDescriptorKeys.Provider) == target.Provider
            && ArtifactDescriptor.Get(set.Descriptor, ArtifactDescriptorKeys.Target) == device.CompilationTarget;

    private CompilationCache SetFor(IComputeDevice device, ArtifactTarget target)
    {
        if (!_sets.TryGetValue(target.Id, out var set))
        {
            var fallbacks = stored is null ? [] : Matching(stored, device, target).Select(s => s.Entries);
            set = new CompilationCache(target.Descriptor, fallbacks);
            _sets[target.Id] = set;
            _order.Add(set);
        }

        return set;
    }
}
