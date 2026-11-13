namespace Sharplight.Runtime;

/// <summary>A device an <see cref="IExecutionProvider"/> can run on.</summary>
public sealed record DeviceInfo
{
    public required string Provider { get; init; }
    /// <summary>Provider-local index (stable for the process lifetime).</summary>
    public required int Index { get; init; }
    public required string Name { get; init; }
    public required DeviceKind Kind { get; init; }
    /// <summary>Device-local memory in bytes (system RAM for CPU devices).</summary>
    public long TotalMemory { get; init; }
    public string Vendor { get; init; } = "";
    public string Driver { get; init; } = "";
    /// <summary>Stable identifier across providers (e.g. PCI bus id / device UUID) so the same GPU
    /// exposed by several providers (Vulkan + CUDA) can be recognized.</summary>
    public string? Uuid { get; init; }
    /// <summary>
    /// Optional hardware/driver features detected on this device, using provider-neutral names from
    /// <see cref="DeviceFeatures"/>. Providers only use a feature when it is detected AND not disabled
    /// via <see cref="DeviceOptions.DisabledFeatures"/> / <see cref="CompileOptions.DisabledFeatures"/>;
    /// otherwise they fall back to portable code paths.
    /// </summary>
    public IReadOnlySet<string> Features { get; init; } = new HashSet<string>();
    /// <summary>
    /// Longest stretch of GPU work (ms) the runtime should hand to the device without letting it go idle and be
    /// scheduled again; 0 = no limit. Devices that also drive a display or share memory with the host (integrated GPUs)
    /// can stall, be reset by a watchdog or lose the device when one submission runs for seconds, so a provider splits
    /// long executions (large prefill batches, big vision encodes) into submissions of about this duration.
    /// Overridable per device with <see cref="DeviceOptions.Settings"/> key <c>submission_budget_ms</c>
    /// and process-wide with the environment variable <c>SHARPLIGHT_SUBMISSION_BUDGET_MS</c> (0 turns it off).
    /// </summary>
    public double SubmissionBudgetMs { get; init; }
    /// <summary>Provider-specific informational details (API version, raw extension info, limits, ...).</summary>
    public IReadOnlyDictionary<string, string> Capabilities { get; init; } = new Dictionary<string, string>();

    public override string ToString() => $"{Provider}:{Index} {Name} ({Kind}, {TotalMemory / (1024 * 1024)} MiB)";
}
