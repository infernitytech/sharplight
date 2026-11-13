using Sharplight.Graph;

namespace Sharplight.Runtime;

public enum DeviceKind { DiscreteGpu, IntegratedGpu, VirtualGpu, Cpu, Other }

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

public sealed record ProviderAvailability(bool IsAvailable, string Reason)
{
    public static ProviderAvailability Available(string detail = "") => new(true, detail);
    public static ProviderAvailability Unavailable(string reason) => new(false, reason);
}

/// <summary>Options when opening a device.</summary>
public sealed class DeviceOptions
{
    /// <summary>Max bytes of device memory the runtime may use (null = most of the free memory).</summary>
    public long? MemoryBudget { get; set; }
    /// <summary>Enable API validation / debug checks (slow).</summary>
    public bool Debug { get; set; }
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

/// <summary>
/// Provider-neutral names of optional device features. A provider maps its own mechanisms onto these
/// (e.g. Vulkan VK_KHR_cooperative_matrix, CUDA tensor cores / mma.sync, Metal simdgroup_matrix all
/// provide <see cref="MatrixF16"/>). Hosts can turn any of them off for a device or a single compilation.
/// </summary>
public static class DeviceFeatures
{
    /// <summary>Subgroup/warp-level matrix multiply-accumulate on f16 inputs with f32 accumulation (tensor cores).</summary>
    public const string MatrixF16 = "matrix_f16";
    /// <summary>Matrix multiply-accumulate on bf16 inputs with f32 accumulation.</summary>
    public const string MatrixBF16 = "matrix_bf16";
    /// <summary>Matrix multiply-accumulate on int8 inputs with int32 accumulation.</summary>
    public const string MatrixInt8 = "matrix_int8";
    /// <summary>
    /// Workgroup-wide f16 matrix multiply-accumulate on large tiles whose operands are loaded straight from memory
    /// by tensor addressing (bounds-clamped slices, transposed views) and decoded on load from block formats
    /// (Vulkan: VK_NV_cooperative_matrix2). Lets the device stage tiles itself instead of the kernel.
    /// </summary>
    public const string MatrixWorkgroup = "matrix_workgroup";
    /// <summary>Native 16-bit float arithmetic in kernels.</summary>
    public const string Float16 = "fp16";
    /// <summary>Native bfloat16 types in kernels.</summary>
    public const string BFloat16 = "bf16";
    /// <summary>Packed int8 dot-product instructions (dp4a-style).</summary>
    public const string IntDot = "int_dot";
    /// <summary>Subgroup/warp collective operations (reductions, shuffles).</summary>
    public const string SubgroupOps = "subgroup_ops";
    /// <summary>The kernel can pin the subgroup size.</summary>
    public const string SubgroupSizeControl = "subgroup_size_control";

    /// <summary>The device reports a live memory budget / usage that accounts for other processes (Vulkan: VK_EXT_memory_budget); without it memory figures are estimates from our own allocations.</summary>
    public const string MemoryBudget = "memory_budget";

    public static IReadOnlyList<string> All { get; } = [MemoryBudget, MatrixF16, MatrixBF16, MatrixInt8, MatrixWorkgroup, Float16, BFloat16, IntDot, SubgroupOps, SubgroupSizeControl];

    /// <summary>
    /// Features disabled process-wide through the environment variable <c>SHARPLIGHT_DISABLE_FEATURES</c>
    /// (comma-separated names, or "all"), applied on top of the per-device/per-compile options.
    /// </summary>
    public static IReadOnlySet<string> DisabledByEnvironment { get; } = ParseEnv();

    /// <summary>Effective feature set: detected minus every disable list.</summary>
    public static IReadOnlySet<string> Effective(IReadOnlySet<string> detected, params IEnumerable<string>?[] disabled)
    {
        var set = new HashSet<string>(detected, StringComparer.OrdinalIgnoreCase);
        set.ExceptWith(DisabledByEnvironment);
        foreach (var d in disabled) if (d is not null) set.ExceptWith(d);
        return set;
    }

    static HashSet<string> ParseEnv()
    {
        var v = Environment.GetEnvironmentVariable("SHARPLIGHT_DISABLE_FEATURES");
        var set = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        if (string.IsNullOrWhiteSpace(v)) return set;
        foreach (var f in v.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
            if (f.Equals("all", StringComparison.OrdinalIgnoreCase)) set.UnionWith(All); else set.Add(f);
        return set;
    }
}

/// <summary>
/// A backend (Vulkan, CUDA, CPU, Metal, ...). Providers are cheap to construct; they only touch the
/// driver when <see cref="CheckAvailability"/> / <see cref="GetDevices"/> are called.
/// </summary>
public interface IExecutionProvider : IDisposable
{
    string Name { get; }
    /// <summary>Detects whether the provider can run on this machine (driver/library present, ...).</summary>
    ProviderAvailability CheckAvailability();
    IReadOnlyList<DeviceInfo> GetDevices();
    IComputeDevice OpenDevice(DeviceInfo device, DeviceOptions? options = null);
}

public readonly record struct OpSupport(bool IsSupported, string? Reason = null)
{
    public static OpSupport Yes => new(true);
    public static OpSupport No(string reason) => new(false, reason);
}

/// <summary>Memory figures of an opened device, in bytes.</summary>
/// <param name="Total">Physical capacity (device-local heap; system RAM for CPU).</param>
/// <param name="Budget">Most this process may use in total: the smaller of the configured cap and what the driver reports
/// as obtainable (which shrinks when other applications use the device).</param>
/// <param name="Usage">What this process uses now, as reported by the driver when <paramref name="IsExact"/>, else our own tally.</param>
/// <param name="UsedByUs">Bytes Sharplight allocated on this device instance (weights, state, activations, staging).</param>
/// <param name="IsExact">True when <paramref name="Budget"/>/<paramref name="Usage"/> come from the driver.</param>
public readonly record struct DeviceMemoryInfo(long Total, long Budget, long Usage, long UsedByUs, bool IsExact)
{
    /// <summary>Bytes this process can still allocate.</summary>
    public long Available => Math.Max(0, Budget - Usage);
    public override string ToString() => $"total {Total / 1048576} MiB, budget {Budget / 1048576} MiB, used {Usage / 1048576} MiB (ours {UsedByUs / 1048576} MiB), available {Available / 1048576} MiB{(IsExact ? "" : " (estimate)")}";
}

/// <summary>An opened device able to compile and run graph segments.</summary>
public interface IComputeDevice : IDisposable
{
    DeviceInfo Info { get; }
    /// <summary>Features this opened device may use: detected minus <see cref="DeviceOptions.DisabledFeatures"/> (and the environment).</summary>
    IReadOnlySet<string> EnabledFeatures => DeviceFeatures.Effective(Info.Features);
    /// <summary>Memory budget remaining for new allocations (see <see cref="GetMemoryInfo"/>).</summary>
    long AvailableMemory { get; }
    /// <summary>
    /// Current device memory figures. Devices that can ask the driver (<see cref="DeviceFeatures.MemoryBudget"/>) report
    /// the live budget, which already accounts for other processes; others derive it from <see cref="AvailableMemory"/>.
    /// Cheap enough to call per load decision; safe from any thread.
    /// </summary>
    DeviceMemoryInfo GetMemoryInfo()
    {
        long total = Info.TotalMemory, avail = AvailableMemory;
        return new DeviceMemoryInfo(total, total, Math.Max(0, total - avail), Math.Max(0, total - avail), false);
    }
    OpSupport Supports(Node node);
    ICompiledSegment Compile(GraphSegment segment, CompileOptions options);

    /// <summary>
    /// Identifies devices that can share compilation artifacts (provider, device model, driver, enabled features).
    /// Saved compiled models keep one <see cref="ICompilationCache"/> per target.
    /// </summary>
    string CompilationTarget => $"{Info.Provider}|{Info.Name}|{Info.Driver}|{string.Join(",", EnabledFeatures.Order(StringComparer.Ordinal))}";

    /// <summary>
    /// Adds device-level compilation state (e.g. a driver pipeline cache covering every kernel compiled so far) to
    /// <paramref name="cache"/> before it is saved.
    /// </summary>
    void ExportCompilationState(ICompilationCache cache) { }
}

/// <summary>Bounds for compilation: buffers are sized for the maximum and kernels specialized.</summary>
public sealed class CompileOptions
{
    /// <summary>Maximum tokens per execution (bound for the symbolic token dimension <c>T</c>).</summary>
    public long MaxTokens { get; init; } = 512;
    /// <summary>Context length (bound for the <c>ctx</c> symbol of caches).</summary>
    public long ContextLength { get; init; } = 4096;
    /// <summary>Additional symbol bounds.</summary>
    public IReadOnlyDictionary<string, long> SymbolBounds { get; init; } = new Dictionary<string, long>();
    /// <summary>
    /// Device features (<see cref="DeviceFeatures"/>) this compilation must not use, on top of the
    /// device-level disables — e.g. to A/B test or work around a driver bug for one model.
    /// </summary>
    public IReadOnlySet<string> DisabledFeatures { get; init; } = new HashSet<string>();

    /// <summary>
    /// Allow kernels to compute f32-weight ops at reduced precision (e.g. f16 tensor-core matmuls with
    /// f32 accumulation). Ops on f16/bf16/quantized weights may always use such paths, since their
    /// error is below the weights' own precision.
    /// </summary>
    public bool AllowReducedPrecision { get; init; }

    /// <summary>
    /// Compilation cache to use on a device (kernel binaries, pipeline caches): consulted before compiling and
    /// filled with everything the compilation used. Null = the provider's own per-user caches only.
    /// </summary>
    public Func<IComputeDevice, ICompilationCache?>? CacheFor { get; init; }

    /// <summary>Features a compilation on <paramref name="device"/> may use.</summary>
    public IReadOnlySet<string> FeaturesFor(IComputeDevice device) => DeviceFeatures.Effective(device.EnabledFeatures, DisabledFeatures);

    public long Bound(string symbol) => symbol switch
    {
        Models.WellKnown.TokenSymbol => MaxTokens,
        Models.WellKnown.ContextSymbol => ContextLength,
        _ => SymbolBounds.TryGetValue(symbol, out var v) ? v : throw new InvalidOperationException($"No bound for symbol '{symbol}'."),
    };
}

/// <summary>Per-execution parameters.</summary>
public sealed class RunParameters
{
    /// <summary>Bindings of symbolic dimensions (e.g. T = number of tokens).</summary>
    public Dictionary<string, long> Symbols { get; } = new(StringComparer.Ordinal);
    /// <summary>Runtime scalars (e.g. n_past).</summary>
    public Dictionary<string, long> Scalars { get; } = new(StringComparer.Ordinal);
}

/// <summary>A compiled, device-resident part of a graph.</summary>
public interface ICompiledSegment : IDisposable
{
    GraphSegment Segment { get; }
    /// <summary>Uploads host data for a segment input (dense, logical layout).</summary>
    void SetInput(Value value, ReadOnlySpan<byte> data);
    /// <summary>Records/executes the segment. Implementations may run asynchronously until an output is read.</summary>
    void Execute(RunParameters parameters);
    /// <summary>Downloads a segment output (dense, logical layout) after <see cref="Execute"/>.</summary>
    void GetOutput(Value value, Span<byte> destination);
    /// <summary>Clears persistent state (KV caches, recurrent states) owned by this segment.</summary>
    void ResetState();
    /// <summary>Device memory held by this segment (weights, state, activations).</summary>
    long DeviceMemoryBytes { get; }
}
