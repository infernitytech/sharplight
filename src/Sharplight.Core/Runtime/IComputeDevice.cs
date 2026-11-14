using Sharplight.Graph;

namespace Sharplight.Runtime;

/// <summary>An opened device able to compile and run graph segments.</summary>
public interface IComputeDevice : IDisposable
{
    /// <summary>The device this instance runs on.</summary>
    public DeviceInfo Info { get; }
    /// <summary>Features this opened device may use: detected minus <see cref="DeviceOptions.DisabledFeatures"/> (and the environment).</summary>
    public IReadOnlySet<string> EnabledFeatures => DeviceFeatures.Effective(Info.Features);
    /// <summary>Memory budget remaining for new allocations (see <see cref="GetMemoryInfo"/>).</summary>
    public long AvailableMemory { get; }
    /// <summary>
    /// Current device memory figures. Devices that can ask the driver (<see cref="DeviceFeatures.MemoryBudget"/>) report
    /// the live budget, which already accounts for other processes; others derive it from <see cref="AvailableMemory"/>.
    /// Cheap enough to call per load decision; safe from any thread.
    /// </summary>
    public DeviceMemoryInfo GetMemoryInfo()
    {
        long total = Info.TotalMemory;
        long avail = AvailableMemory;
        return new DeviceMemoryInfo(total, total, Math.Max(0, total - avail), Math.Max(0, total - avail), false);
    }
    /// <summary>
    /// Whether compilations on this device share weight buffers through <see cref="CompileOptions.SharedWeights"/> (see
    /// <see cref="Runtime.SharedWeights"/> for the provider contract). False (the default): each compiled segment holds its own copy.
    /// </summary>
    public bool SharesWeights => false;
    /// <summary>Whether the device can execute the node; unsupported nodes go to the placement fallback.</summary>
    public OpSupport Supports(Node node);
    /// <summary>
    /// Compiles a segment for this device, allocating weights, state and activation memory.
    /// </summary>
    /// <param name="segment">The graph segment to compile.</param>
    /// <param name="options">Bounds for symbolic dimensions and feature switches.</param>
    /// <returns>The compiled segment; the caller disposes it.</returns>
    public ICompiledSegment Compile(GraphSegment segment, CompileOptions options);

    /// <summary>
    /// Version 1 identifier of devices that can share compilation artifacts (provider, device model, driver, enabled features).
    /// Only used to find legacy kernel sets (<see cref="ArtifactTiers.Mixed"/>) stored for this device; new artifacts are
    /// described by <see cref="ArtifactTargets"/>.
    /// </summary>
    public string CompilationTarget => $"{Info.Provider}|{Info.Name}|{Info.Driver}|{string.Join(",", EnabledFeatures.Order(StringComparer.Ordinal))}";

    /// <summary>
    /// The tiers of precompiled artifacts this device produces and consumes (e.g. a code tier keyed by a device profile and a
    /// binary tier keyed by device and driver), in routing order: a cache key belongs to the first target that claims it
    /// (<see cref="ArtifactTarget.KeyPrefixes"/>). Empty when the device has nothing worth precompiling.
    /// </summary>
    public IReadOnlyList<ArtifactTarget> ArtifactTargets => [];

    /// <summary>
    /// Adds device-level compilation state (e.g. a driver pipeline cache covering every kernel compiled so far) to
    /// <paramref name="cache"/> before it is saved.
    /// </summary>
    public void ExportCompilationState(ICompilationCache cache) { }

    /// <summary>
    /// What code generation on this opened device depends on (<see cref="DeviceProfile"/>), so code for it can be generated
    /// elsewhere through an <see cref="IOfflineCompilationProvider"/>; <see langword="null"/> when the provider cannot compile
    /// offline. Its features are <see cref="EnabledFeatures"/>.
    /// </summary>
    public DeviceProfile? CreateProfile() => null;
}
