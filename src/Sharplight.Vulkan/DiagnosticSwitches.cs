using System.Globalization;

namespace Sharplight.Vulkan;

/// <summary>
/// Every <c>SHARPLIGHT_*</c> environment switch read by the Vulkan project. All of them are diagnostics; the defaults
/// are the shipped behavior.
/// <list type="bullet">
/// <item><c>SHARPLIGHT_VK_VALIDATION=1</c>: enable the Khronos validation layer (GPU-assisted validation is configured
/// through the layer's own environment variables).</item>
/// <item><c>SHARPLIGHT_VK_PROFILE=1|2</c>: record per-dispatch GPU timestamps and accumulate them in
/// <see cref="VulkanSegment.ProfileData"/>; <c>2</c> additionally splits the entries by kernel entry point.</item>
/// <item><c>SHARPLIGHT_VK_PIPELINE_STATS=1</c>: print the driver's statistics of every compute pipeline (registers, shared
/// memory, spills where the driver reports them) to stderr, through VK_KHR_pipeline_executable_properties.</item>
/// <item><c>SHARPLIGHT_VK_TRACE=1</c>: log every command buffer recording and submission to stderr.</item>
/// <item><c>SHARPLIGHT_VK_NO_BARRIERS=1</c>: drop inter-dispatch barriers (wrong results) to measure their cost.</item>
/// <item><c>SHARPLIGHT_VK_REORDER=0</c>: record launches in program order instead of filling each barrier batch with later
/// independent launches (<see cref="HazardTracker.Order"/>).</item>
/// <item><c>SHARPLIGHT_VK_MAX_LAUNCHES=n</c>: record only the first <c>n</c> launches (bisecting a faulting kernel).</item>
/// <item><c>SHARPLIGHT_VK_LIST_LAUNCHES=1</c>: print each recorded launch.</item>
/// <item><c>SHARPLIGHT_VK_SPIRV_PORTABILITY=0</c>: on portability-subset devices (MoltenVK), create pipelines from the
/// kernels as compiled instead of applying <see cref="SpirvPortability.Rewrite"/>; <c>=all</c> applies the rewrite on
/// every device (to check the rewritten kernels on a native driver).</item>
/// <item><c>SHARPLIGHT_LOAD_TIMING=1</c>: print load-phase timings (pipeline creation, upload) to stderr.</item>
/// <item><c>SHARPLIGHT_SUBMISSION_BUDGET_MS=ms</c>: overrides the per-device submission budget (0 disables slicing); the
/// <c>submission_budget_ms</c> device setting wins over it.</item>
/// </list>
/// </summary>
internal static class DiagnosticSwitches
{
    /// <summary>Enables the validation layer for instances created afterwards (default of <see cref="VulkanProvider.EnableValidation"/>).</summary>
    public static bool Validation { get; } = IsSet("SHARPLIGHT_VK_VALIDATION");

    /// <summary>True when per-dispatch profiling is on by default (<c>SHARPLIGHT_VK_PROFILE</c> is 1 or 2).</summary>
    public static bool Profile { get; } = Read("SHARPLIGHT_VK_PROFILE") is "1" or "2";

    /// <summary>True when profile entries are keyed by kernel entry point too (<c>SHARPLIGHT_VK_PROFILE=2</c>).</summary>
    public static bool ProfileDetail { get; } = Read("SHARPLIGHT_VK_PROFILE") == "2";

    /// <summary>Prints pipeline statistics as pipelines are created.</summary>
    public static bool PipelineStats { get; } = IsSet("SHARPLIGHT_VK_PIPELINE_STATS");

    /// <summary>Logs command buffer recording and submission.</summary>
    public static bool Trace { get; } = IsSet("SHARPLIGHT_VK_TRACE");

    /// <summary>Drops barriers between dispatches (results become wrong).</summary>
    public static bool NoBarriers { get; } = IsSet("SHARPLIGHT_VK_NO_BARRIERS");

    /// <summary>True when recording may move independent launches into earlier barrier batches.</summary>
    public static bool ReorderLaunches { get; } = Read("SHARPLIGHT_VK_REORDER") != "0";

    /// <summary>Number of launches recorded per command buffer; <see cref="int.MaxValue"/> records all.</summary>
    public static int MaxLaunches { get; } = int.TryParse(Read("SHARPLIGHT_VK_MAX_LAUNCHES"), out int max) ? max : int.MaxValue;

    /// <summary>Prints each recorded launch.</summary>
    public static bool ListLaunches { get; } = IsSet("SHARPLIGHT_VK_LIST_LAUNCHES");

    /// <summary>True when kernels for portability-subset devices get <see cref="SpirvPortability.Rewrite"/> (the default).</summary>
    public static bool PortabilityRewrite { get; } = Read("SHARPLIGHT_VK_SPIRV_PORTABILITY") != "0";

    /// <summary>True when <see cref="SpirvPortability.Rewrite"/> applies on every device (<c>SHARPLIGHT_VK_SPIRV_PORTABILITY=all</c>).</summary>
    public static bool PortabilityRewriteAll { get; } = Read("SHARPLIGHT_VK_SPIRV_PORTABILITY") == "all";

    /// <summary>Prints load-phase timings.</summary>
    public static bool LoadTiming { get; } = IsSet("SHARPLIGHT_LOAD_TIMING");

    /// <summary>
    /// The submission budget in milliseconds from the environment, or null when unset or malformed. Read on every call
    /// because it is applied per opened device.
    /// </summary>
    public static double? SubmissionBudgetMs(string? deviceSetting)
    {
        string? text = deviceSetting ?? Read("SHARPLIGHT_SUBMISSION_BUDGET_MS");
        return text is not null && double.TryParse(text, CultureInfo.InvariantCulture, out double ms) ? ms : null;
    }

    private static string? Read(string name) => Environment.GetEnvironmentVariable(name);

    private static bool IsSet(string name) => Read(name) == "1";
}
