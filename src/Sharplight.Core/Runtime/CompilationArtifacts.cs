namespace Sharplight.Runtime;

/// <summary>
/// The precompiled artifacts of one model across devices: gives each device a cache (<see cref="For"/>, to use as
/// <see cref="CompileOptions.CacheFor"/>) that is seeded from the matching stored sets (device code of its device and driver,
/// portable code of its profile) and records the portable code compilation uses, so it can be exported (<see cref="Export"/>,
/// <see cref="PortableCodeFor"/>), compiled into device code (<see cref="CompileDeviceCodeAsync"/>) and stored with the model.
/// Thread-safe.
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
    /// The non-empty artifact sets recorded so far: the portable code (<see cref="ArtifactTiers.Code"/>) every device generated or
    /// read while compiling. Driver binaries are not recorded while compiling; produce them with <see cref="CompileDeviceCodeAsync"/>
    /// (the device-specific stage of each device's provider).
    /// </summary>
    /// <param name="tiers">Only sets of these tiers (<see cref="ArtifactTiers"/>), or <see langword="null"/> for all.</param>
    public IReadOnlyList<CompilationCache> Export(IReadOnlySet<string>? tiers = null)
    {
        lock (_lock)
        {
            return [.. _order.Where(s => s.Entries.Count > 0 && (tiers is null || tiers.Contains(ArtifactDescriptor.Get(s.Descriptor, ArtifactDescriptorKeys.Tier))))];
        }
    }

    /// <summary>
    /// The portable code <paramref name="device"/> compiled with so far (its <see cref="ArtifactTiers.Code"/> set, shared with every device
    /// of the same profile); empty when the device produces no portable code or compiled nothing with these artifacts.
    /// </summary>
    /// <param name="device">A device that compiled with <see cref="For"/>.</param>
    public PortableCodeSet PortableCodeFor(IComputeDevice device)
    {
        ArgumentNullException.ThrowIfNull(device);
        lock (_lock)
        {
            var code = device.ArtifactTargets.FirstOrDefault(t => t.Tier == ArtifactTiers.Code);
            if (!_devices.ContainsKey(device) || code is null || !_sets.TryGetValue(code.Id, out var set))
            {
                return PortableCodeSet.Empty(device.Info.Provider);
            }

            return PortableCodeSet.FromArtifactSet(set);
        }
    }

    /// <summary>
    /// Runs the device-specific stage (<see cref="IExecutionProvider.CompileDeviceSpecificAsync"/>) for every device that compiled with
    /// these artifacts, on the portable code it used, through the provider among <paramref name="providers"/> it belongs to. Devices
    /// that would produce the same set (identical GPUs on the same driver) are compiled once.
    /// </summary>
    /// <param name="providers">The providers of the devices; devices of other providers are skipped.</param>
    /// <param name="cancellationToken">Cancels between kernels.</param>
    /// <returns>The non-empty device code sets.</returns>
    public async Task<IReadOnlyList<DeviceCodeSet>> CompileDeviceCodeAsync(IEnumerable<IExecutionProvider> providers, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(providers);
        var byName = providers.GroupBy(p => p.Name, StringComparer.Ordinal).ToDictionary(g => g.Key, g => g.First(), StringComparer.Ordinal);
        IComputeDevice[] devices;
        lock (_lock)
        {
            devices = [.. _devices.Where(d => d.Value is not null).Select(d => d.Key)];
        }

        var produced = new Dictionary<string, DeviceCodeSet>(StringComparer.Ordinal);
        foreach (var device in devices)
        {
            var identity = DeviceCodeIdentity(device);
            if (!byName.TryGetValue(device.Info.Provider, out var provider) || identity.Length == 0 || produced.ContainsKey(identity))
            {
                continue;
            }

            produced[identity] = await provider.CompileDeviceSpecificAsync(device, PortableCodeFor(device), cancellationToken).ConfigureAwait(false);
        }

        return [.. produced.Values.Where(s => !s.IsEmpty)];
    }

    /// <summary>
    /// What decides whether two devices produce interchangeable device code: the ids of their <see cref="ArtifactTiers.Binary"/> targets
    /// (device model, driver build, kernel ABI). Empty when the device produces no device code.
    /// </summary>
    /// <param name="device">An opened device.</param>
    public static string DeviceCodeIdentity(IComputeDevice device)
    {
        ArgumentNullException.ThrowIfNull(device);
        return string.Join("+", device.ArtifactTargets.Where(t => t.Tier == ArtifactTiers.Binary).Select(t => t.Id).Order(StringComparer.Ordinal));
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
