using System.Globalization;

namespace Sharplight.Cuda;

/// <summary>
/// Environment switches of the CUDA provider (diagnostics and comparisons), read once at startup:
/// <list type="bullet">
/// <item><c>SHARPLIGHT_CUDA_RUNTIME_DIR=dir</c>: directory with NVRTC, its builtins library and the floating-point headers, searched
/// before the application directory (e.g. an unpacked <c>Sharplight.Cuda.Runtime</c> package).</item>
/// <item><c>SHARPLIGHT_CUDA_DUMP_SOURCE=dir</c>: writes the CUDA C++ and the PTX of every kernel compiled with NVRTC into <c>dir</c>.</item>
/// <item><c>SHARPLIGHT_CUDA_TRACE=1</c>: prints module loads and recordings.</item>
/// <item><c>SHARPLIGHT_CUDA_PROFILE=1</c>: times every launch with events and accumulates per-kernel GPU times.</item>
/// <item><c>SHARPLIGHT_CUDA_MAX_SHARED=bytes</c>: shared memory per block the kernels are lowered for (default: the device's opt-in
/// maximum; 49152 keeps CUDA's static limit, the size the Vulkan provider has).</item>
/// <item><c>SHARPLIGHT_CUDA_SYNC=1</c>: waits for the stream after every launch, so a faulting kernel is reported by name.</item>
/// <item><c>SHARPLIGHT_CUDA_CONCURRENT=0</c>: captured step graphs keep one launch after the other instead of running the launches between
/// two hazards (the Vulkan provider's barrier batches) side by side.</item>
/// <item><c>SHARPLIGHT_LOAD_TIMING=1</c>, <c>SHARPLIGHT_SUBMISSION_BUDGET_MS</c>, <c>SHARPLIGHT_CONCURRENT_WORKGROUPS</c>,
/// <c>SHARPLIGHT_DEVICE_FORWARDING=0</c>: as for the Vulkan provider.</item>
/// </list>
/// The GPUs the provider sees are those the driver exposes: <c>CUDA_VISIBLE_DEVICES</c> selects them.
/// </summary>
internal static class DiagnosticSwitches
{
    public static string? DumpSourceDirectory { get; } = Read("SHARPLIGHT_CUDA_DUMP_SOURCE");

    public static bool Trace { get; } = IsSet("SHARPLIGHT_CUDA_TRACE");

    public static bool Profile { get; } = IsSet("SHARPLIGHT_CUDA_PROFILE");

    /// <summary><c>SHARPLIGHT_CUDA_MIN_BLOCKS=file:entry:blocks[;...]</c>: launch bounds for matching kernels (see NvrtcTranslator.WithMinBlocks).</summary>
    public static string? MinBlocks { get; } = Read("SHARPLIGHT_CUDA_MIN_BLOCKS");

    /// <summary>
    /// <c>SHARPLIGHT_CUDA_PDL_TRIGGER=0</c>: kernels compiled for dependent launches only wait for the previous launch at their entry and
    /// do not let the next launch start early (<c>griddepcontrol.launch_dependents</c>); it then starts when this kernel's blocks exit.
    /// </summary>
    public static bool DependentTrigger { get; } = Read("SHARPLIGHT_CUDA_PDL_TRIGGER") != "0";

    public static long? MaxSharedMemory { get; } = Read("SHARPLIGHT_CUDA_MAX_SHARED") is { } v ? long.Parse(v, System.Globalization.CultureInfo.InvariantCulture) : null;

    /// <summary><c>SHARPLIGHT_CUDA_CONCURRENT</c> (default on).</summary>
    public static bool ConcurrentLaunches { get; } = Read("SHARPLIGHT_CUDA_CONCURRENT") != "0";

    /// <summary><c>SHARPLIGHT_CUDA_SYNC=1</c>.</summary>
    public static bool SyncEachLaunch { get; } = IsSet("SHARPLIGHT_CUDA_SYNC");

    public static bool LoadTiming { get; } = IsSet("SHARPLIGHT_LOAD_TIMING");

    /// <summary>
    /// <c>SHARPLIGHT_CUDA_NVRTC_SASS</c>: which kernels NVRTC also compiles to machine code (see NvrtcTranslator.CompilesSass): unset, the
    /// kernels that ask for it; <c>0</c> none (the driver compiles every kernel's PTX); <c>all</c> every kernel; <c>if:text</c> the kernels whose source contains the text
    /// (<c>if:a|b</c>: one of the texts).
    /// </summary>
    public static string? NvrtcSass { get; } = Read("SHARPLIGHT_CUDA_NVRTC_SASS");

    public static bool DeviceForwarding { get; } = Read("SHARPLIGHT_DEVICE_FORWARDING") is not ("0" or "false");

    /// <summary>The submission budget in milliseconds from the device setting or the environment, or null when unset or malformed.</summary>
    public static double? SubmissionBudgetMs(string? deviceSetting)
    {
        string? text = deviceSetting ?? Read("SHARPLIGHT_SUBMISSION_BUDGET_MS");
        return text is not null && double.TryParse(text, CultureInfo.InvariantCulture, out double ms) ? ms : null;
    }

    /// <summary>The concurrent-workgroup hint from the device setting or the environment, or null when unset or malformed.</summary>
    public static int? ConcurrentWorkgroups(string? deviceSetting)
    {
        string? text = deviceSetting ?? Read("SHARPLIGHT_CONCURRENT_WORKGROUPS");
        return text is not null && int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int workgroups) ? workgroups : null;
    }

    private static string? Read(string name) => Environment.GetEnvironmentVariable(name) is { Length: > 0 } value ? value : null;

    private static bool IsSet(string name) => Read(name) == "1";
}
