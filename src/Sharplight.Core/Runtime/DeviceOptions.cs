using System.Globalization;

namespace Sharplight.Runtime;

/// <summary>Options when opening a device.</summary>
public sealed class DeviceOptions
{
    /// <summary>Max bytes of device memory the runtime may use (null = most of the free memory).</summary>
    public long? MemoryBudget { get; set; }
    /// <summary>
    /// Directory of the persistent compilation cache (null = the per-user default, or what <c>SHARPLIGHT_COMPILATION_CACHE</c> says; see
    /// <see cref="FileCompilationCache"/>).
    /// </summary>
    public string? KernelCacheDirectory { get; set; }
    /// <summary>
    /// Use the persistent compilation cache (compiled kernels and driver pipeline caches kept across processes). Default true; hosts
    /// switch it off to never read or write cache files.
    /// </summary>
    public bool UseCompilationCache { get; set; } = true;
    /// <summary>
    /// Features (see <see cref="DeviceFeatures"/>) the device must not use, even if detected. Disabled
    /// features are not enabled in the driver at all (e.g. no extension is requested).
    /// </summary>
    public ISet<string> DisabledFeatures { get; } = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
    /// <summary>
    /// Caps on the device's memory below the hardware's (device heap, host-visible device memory, largest allocation, pinned host
    /// memory), combined with the <c>SHARPLIGHT_LIMIT_*</c> environment variables; see <see cref="DeviceMemoryLimits"/>.
    /// </summary>
    public DeviceMemoryLimits MemoryLimits { get; } = new();
    /// <summary>Free-form provider settings (e.g. CPU "threads").</summary>
    public Dictionary<string, string> Settings { get; } = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// The submission budget in milliseconds of the device: <paramref name="detected"/> unless the <see cref="Settings"/> key
    /// <c>submission_budget_ms</c> or, without it, <c>SHARPLIGHT_SUBMISSION_BUDGET_MS</c> overrides it (negative values clamp to 0).
    /// </summary>
    /// <param name="detected">The budget the provider detected for the device.</param>
    public double ResolveSubmissionBudgetMs(double detected)
    {
        string? text = Setting("submission_budget_ms", "SHARPLIGHT_SUBMISSION_BUDGET_MS");
        return text is not null && double.TryParse(text, CultureInfo.InvariantCulture, out double ms) ? Math.Max(0, ms) : detected;
    }

    /// <summary>
    /// The concurrent-workgroup hint of the device: <paramref name="detected"/> unless the <see cref="Settings"/> key
    /// <c>concurrent_workgroups</c> or, without it, <c>SHARPLIGHT_CONCURRENT_WORKGROUPS</c> overrides it (negative values clamp to 0).
    /// </summary>
    /// <param name="detected">The hint the provider detected for the device.</param>
    public int ResolveConcurrentWorkgroups(int detected)
    {
        string? text = Setting("concurrent_workgroups", "SHARPLIGHT_CONCURRENT_WORKGROUPS");
        return text is not null && int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int workgroups) ? Math.Max(0, workgroups) : detected;
    }

    private string? Setting(string key, string environmentVariable)
        => Settings.TryGetValue(key, out var setting) ? setting
            : Environment.GetEnvironmentVariable(environmentVariable) is { Length: > 0 } value ? value : null;
}
