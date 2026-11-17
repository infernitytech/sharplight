using System.Globalization;

namespace Sharplight.Metal;

/// <summary>
/// Environment switches of the Metal provider (diagnostics and comparisons), read once at startup unless noted:
/// <list type="bullet">
/// <item><c>SHARPLIGHT_MTL_SERIAL=1</c>: encode with a serial dispatch type (every dispatch waits for the previous one) instead of
/// concurrent dispatches with the barriers of the hazard tracker; isolates missing barriers.</item>
/// <item><c>SHARPLIGHT_MTL_FAST_MATH=1</c>: compile kernels with Metal fast math (off by default: it reassociates sums, which breaks
/// row invariance, and assumes finite values, which masked softmax scores are not).</item>
/// <item><c>SHARPLIGHT_MTL_TRACE=1</c>: prints recordings and pipeline creation.</item>
/// <item><c>SHARPLIGHT_MTL_PROFILE=1</c>: runs every launch in its own command buffer and accumulates per-kernel GPU times.</item>
/// <item><c>SHARPLIGHT_MTL_NO_COPY=0</c>: copy the weights into Metal buffers instead of wrapping each mapped model file once with
/// <c>newBufferWithBytesNoCopy</c> (on the CI's Apple GPU, warm page cache: 56 ms against 301 ms for the 0.49 GB smoke model).</item>
/// <item><c>SHARPLIGHT_MTL_DUMP_SOURCE=dir</c>: writes the MSL of every kernel a pipeline is created for into <c>dir</c>.</item>
/// <item><c>SHARPLIGHT_LOAD_TIMING=1</c>, <c>SHARPLIGHT_SUBMISSION_BUDGET_MS</c>, <c>SHARPLIGHT_CONCURRENT_WORKGROUPS</c>,
/// <c>SHARPLIGHT_DEVICE_FORWARDING=0</c>: as for the Vulkan provider.</item>
/// </list>
/// </summary>
internal static class DiagnosticSwitches
{
    public static bool NoCopyWeights { get; } = Read("SHARPLIGHT_MTL_NO_COPY") != "0";

    public static bool Serial { get; } = IsSet("SHARPLIGHT_MTL_SERIAL");

    public static bool FastMath { get; } = IsSet("SHARPLIGHT_MTL_FAST_MATH");

    public static bool Trace { get; } = IsSet("SHARPLIGHT_MTL_TRACE");

    public static bool Profile { get; } = IsSet("SHARPLIGHT_MTL_PROFILE");

    public static string? DumpSourceDirectory { get; } = Read("SHARPLIGHT_MTL_DUMP_SOURCE");

    public static bool LoadTiming { get; } = IsSet("SHARPLIGHT_LOAD_TIMING");

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

    private static string? Read(string name) => Environment.GetEnvironmentVariable(name);

    private static bool IsSet(string name) => Read(name) == "1";
}
