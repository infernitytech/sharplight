namespace Sharplight.Cpu;

/// <summary>
/// Every <c>SHARPLIGHT_CPU_*</c> environment switch read by the CPU provider, read once per process. The defaults are the shipped
/// behavior. <c>SHARPLIGHT_CPU_ISA</c> is documented on <see cref="CpuIsa"/>.
/// <list type="bullet">
/// <item><c>SHARPLIGHT_CPU_THREADS=n</c>: degree of parallelism (default: processor count).</item>
/// <item><c>SHARPLIGHT_CPU_SPIN_US=n</c>: microseconds a pool thread spins before sleeping.</item>
/// <item><c>SHARPLIGHT_CPU_NO_FUSION=1</c>: disable elementwise fusion into producing ops.</item>
/// <item><c>SHARPLIGHT_CPU_NO_PREPACK=1</c>: do not keep a packed copy of static f32 matmul weights.</item>
/// <item><c>SHARPLIGHT_CPU_GEMM_MIN_ROWS=n</c>: override the row thresholds from which matmul takes the GEMM path.</item>
/// <item><c>SHARPLIGHT_CPU_NO_PAIR=1|true</c>: lower the gate and up expert projections of a layer separately instead of as one pair.</item>
/// <item><c>SHARPLIGHT_CPU_PROFILE=1|true|min</c>: per-op timing report (<c>min</c>: minimum per op instance).</item>
/// </list>
/// </summary>
internal static class DiagnosticSwitches
{
    /// <summary>True when elementwise fusion is disabled.</summary>
    public static readonly bool NoFusion = Environment.GetEnvironmentVariable("SHARPLIGHT_CPU_NO_FUSION") == "1";

    /// <summary>True when matmul weight prepacking is disabled.</summary>
    public static readonly bool NoPrepack = Environment.GetEnvironmentVariable("SHARPLIGHT_CPU_NO_PREPACK") == "1";

    /// <summary>True when paired expert projections (gate and up of one layer, see <c>PairedMatMulId</c>) are disabled.</summary>
    public static readonly bool NoExpertPairs = Environment.GetEnvironmentVariable("SHARPLIGHT_CPU_NO_PAIR") is "1" or "true";
}
