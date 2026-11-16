namespace Sharplight.Cpu;

/// <summary>
/// Every <c>SHARPLIGHT_CPU_*</c> environment switch read by the CPU provider, read once per process. The defaults are the shipped
/// behavior. <c>SHARPLIGHT_CPU_ISA</c> is documented on <see cref="CpuIsa"/>.
/// <list type="bullet">
/// <item><c>SHARPLIGHT_CPU_THREADS=n</c>: degree of parallelism (default: processor count).</item>
/// <item><c>SHARPLIGHT_CPU_SPIN_US=n</c>: microseconds a pool thread spins before sleeping.</item>
/// <item><c>SHARPLIGHT_CPU_POOL=0</c>: use <see cref="Parallel"/> instead of the spinning thread pool.</item>
/// <item><c>SHARPLIGHT_CPU_AFFINITY=0</c>: one shared work range per loop instead of one range per participant.</item>
/// <item><c>SHARPLIGHT_CPU_NO_FUSION=1</c>: disable elementwise fusion into producing ops.</item>
/// <item><c>SHARPLIGHT_CPU_NO_PREPACK=1</c>: do not keep a packed copy of static f32 matmul weights.</item>
/// <item><c>SHARPLIGHT_CPU_GEMM_MIN_ROWS=n</c>: override the row thresholds from which matmul takes the GEMM path.</item>
/// <item><c>SHARPLIGHT_CPU_NO_SHIFTED_CONV=1</c>: disable the tap-major shifted convolution.</item>
/// <item><c>SHARPLIGHT_CPU_CONV_PLAN=tiles</c>: never choose the weight-major convolution plan.</item>
/// <item><c>SHARPLIGHT_CPU_CL_CHUNKS=0</c>: channels-last convolution runs as one chunk.</item>
/// <item><c>SHARPLIGHT_CPU_PROFILE=1|true|min</c>: per-op timing report (<c>min</c>: minimum per op instance).</item>
/// </list>
/// </summary>
internal static class DiagnosticSwitches
{
    /// <summary>True when the spinning thread pool is enabled.</summary>
    public static readonly bool UsePool = Environment.GetEnvironmentVariable("SHARPLIGHT_CPU_POOL") != "0";

    /// <summary>True when loops use one shared work range.</summary>
    public static readonly bool SharedRanges = Environment.GetEnvironmentVariable("SHARPLIGHT_CPU_AFFINITY") == "0";

    /// <summary>True when elementwise fusion is disabled.</summary>
    public static readonly bool NoFusion = Environment.GetEnvironmentVariable("SHARPLIGHT_CPU_NO_FUSION") == "1";

    /// <summary>True when matmul weight prepacking is disabled.</summary>
    public static readonly bool NoPrepack = Environment.GetEnvironmentVariable("SHARPLIGHT_CPU_NO_PREPACK") == "1";

    /// <summary>True when the shifted convolution is disabled.</summary>
    public static readonly bool NoShiftedConv = Environment.GetEnvironmentVariable("SHARPLIGHT_CPU_NO_SHIFTED_CONV") == "1";

    /// <summary>True when the weight-major convolution plan is disabled.</summary>
    public static readonly bool NoWeightMajorConvPlan = Environment.GetEnvironmentVariable("SHARPLIGHT_CPU_CONV_PLAN") == "tiles";

    /// <summary>True when channels-last convolution chunking is disabled.</summary>
    public static readonly bool NoChannelsLastChunks = Environment.GetEnvironmentVariable("SHARPLIGHT_CPU_CL_CHUNKS") == "0";
}
