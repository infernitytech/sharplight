using Sharplight.Graph;

namespace Sharplight.Runtime;

/// <summary>An opened device able to compile and run graph segments.</summary>
public interface IComputeDevice : IDisposable
{
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
        long total = Info.TotalMemory, avail = AvailableMemory;
        return new DeviceMemoryInfo(total, total, Math.Max(0, total - avail), Math.Max(0, total - avail), false);
    }
    public OpSupport Supports(Node node);
    public ICompiledSegment Compile(GraphSegment segment, CompileOptions options);

    /// <summary>
    /// Identifies devices that can share compilation artifacts (provider, device model, driver, enabled features).
    /// Saved compiled models keep one <see cref="ICompilationCache"/> per target.
    /// </summary>
    public string CompilationTarget => $"{Info.Provider}|{Info.Name}|{Info.Driver}|{string.Join(",", EnabledFeatures.Order(StringComparer.Ordinal))}";

    /// <summary>
    /// Adds device-level compilation state (e.g. a driver pipeline cache covering every kernel compiled so far) to
    /// <paramref name="cache"/> before it is saved.
    /// </summary>
    public void ExportCompilationState(ICompilationCache cache) { }
}
