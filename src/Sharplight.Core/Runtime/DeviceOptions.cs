namespace Sharplight.Runtime;

/// <summary>Options when opening a device.</summary>
public sealed class DeviceOptions
{
    /// <summary>Max bytes of device memory the runtime may use (null = most of the free memory).</summary>
    public long? MemoryBudget { get; set; }
    /// <summary>Directory for compiled kernel caches (null = default per-user cache dir).</summary>
    public string? KernelCacheDirectory { get; set; }
    /// <summary>
    /// Features (see <see cref="DeviceFeatures"/>) the device must not use, even if detected. Disabled
    /// features are not enabled in the driver at all (e.g. no extension is requested).
    /// </summary>
    public ISet<string> DisabledFeatures { get; } = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
    /// <summary>Free-form provider settings (e.g. CPU "threads").</summary>
    public Dictionary<string, string> Settings { get; } = new(StringComparer.OrdinalIgnoreCase);
}
