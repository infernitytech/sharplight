using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace Sharplight.Cpu;

/// <summary>
/// Instruction-set selection of the CPU kernels: the host's tiers are detected once, and the diagnostic environment switch
/// <c>SHARPLIGHT_CPU_ISA</c> (<c>scalar</c>, <c>v128</c>, <c>neon</c>, <c>neondot</c>, <c>avx2</c>, <c>avxvnni</c>, <c>avx512</c>,
/// <c>avx512vnni</c>) caps the
/// tier used by every kernel of the process, to benchmark or test the slower paths on a fast machine. A cap the host does not
/// support selects the highest supported tier below it. Kernels test the <c>Use*</c> properties, which are JIT-time constants.
/// </summary>
internal static class CpuIsa
{
    /// <summary>Name of the environment switch that caps the tier.</summary>
    public const string EnvironmentVariable = "SHARPLIGHT_CPU_ISA";

    private static readonly bool s_v128;
    private static readonly bool s_v256;
    private static readonly bool s_v512;
    private static readonly bool s_avxVnni;
    private static readonly bool s_avx512Vnni;
    private static readonly bool s_neon;
    private static readonly bool s_neonDot;

    static CpuIsa()
    {
        Detected = Highest(CpuIsaLevel.Avx512Vnni);
        Level = Highest(Parse(Environment.GetEnvironmentVariable(EnvironmentVariable)) ?? CpuIsaLevel.Avx512Vnni);
        s_v128 = Level >= CpuIsaLevel.Vector128;
        s_v256 = Level >= CpuIsaLevel.Avx2;
        s_v512 = Level >= CpuIsaLevel.Avx512;
        s_avxVnni = Level == CpuIsaLevel.AvxVnni || (Level == CpuIsaLevel.Avx512Vnni && IsSupported(CpuIsaLevel.AvxVnni));
        s_avx512Vnni = Level == CpuIsaLevel.Avx512Vnni;
        s_neon = Level is CpuIsaLevel.Neon or CpuIsaLevel.NeonDot;
        s_neonDot = Level == CpuIsaLevel.NeonDot;
    }

    /// <summary>The highest tier the host supports.</summary>
    public static CpuIsaLevel Detected { get; }

    /// <summary>The tier the kernels use: <see cref="Detected"/>, capped by <c>SHARPLIGHT_CPU_ISA</c>.</summary>
    public static CpuIsaLevel Level { get; }

    /// <summary>128-bit vector paths are enabled.</summary>
    public static bool Use128
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => s_v128 && Vector128.IsHardwareAccelerated;
    }

    /// <summary>256-bit vector paths (AVX2 + FMA) are enabled.</summary>
    public static bool Use256
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => s_v256 && Avx2.IsSupported && Fma.IsSupported;
    }

    /// <summary>512-bit vector paths (AVX-512 F/BW/VL) are enabled.</summary>
    public static bool Use512
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => s_v512 && Avx512F.IsSupported && Avx512BW.IsSupported && Vector512.IsHardwareAccelerated;
    }

    /// <summary>256-bit VNNI dot products (<c>vpdpbusd ymm</c>) are enabled.</summary>
    public static bool UseAvxVnni
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => s_avxVnni && AvxVnni.IsSupported;
    }

    /// <summary>512-bit VNNI dot products (<c>vpdpbusd zmm</c>) are enabled.</summary>
    public static bool UseAvx512Vnni
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => s_avx512Vnni && AvxVnni.V512.IsSupported && Use512;
    }

    /// <summary>
    /// Arm64 NEON paths are enabled (128-bit vectors with by-lane fused multiply-add). Always false on x86, where the JIT
    /// removes these paths.
    /// </summary>
    public static bool UseNeon
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => s_neon && AdvSimd.Arm64.IsSupported;
    }

    /// <summary>Arm64 int8 dot products (<c>sdot</c>) are enabled.</summary>
    public static bool UseNeonDot
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => s_neonDot && Dp.IsSupported && AdvSimd.Arm64.IsSupported;
    }

    /// <summary>
    /// Whether the kernels have packed int8 dot products on this tier (x86 <c>vpmaddubsw</c> / VNNI, Arm64 <c>sdot</c>), which
    /// makes the quantized-activation matmul (<see cref="Runtime.DeviceFeatures.IntDot"/>) worthwhile.
    /// </summary>
    public static bool HasPackedIntDot => Level >= CpuIsaLevel.Avx2 || Level == CpuIsaLevel.NeonDot;

    /// <summary>Whether the host can run the kernels of <paramref name="level"/>.</summary>
    public static bool IsSupported(CpuIsaLevel level) => level switch
    {
        CpuIsaLevel.Scalar => true,
        CpuIsaLevel.Vector128 => Vector128.IsHardwareAccelerated,
        CpuIsaLevel.Neon => Vector128.IsHardwareAccelerated && AdvSimd.Arm64.IsSupported,
        CpuIsaLevel.NeonDot => IsSupported(CpuIsaLevel.Neon) && Dp.IsSupported,
        CpuIsaLevel.Avx2 => Vector128.IsHardwareAccelerated && Avx2.IsSupported && Fma.IsSupported,
        CpuIsaLevel.AvxVnni => IsSupported(CpuIsaLevel.Avx2) && AvxVnni.IsSupported,
        CpuIsaLevel.Avx512 => IsSupported(CpuIsaLevel.Avx2) && Avx512F.IsSupported && Avx512F.VL.IsSupported && Avx512BW.IsSupported
            && Avx512BW.VL.IsSupported && Avx512DQ.IsSupported && Vector512.IsHardwareAccelerated,
        CpuIsaLevel.Avx512Vnni => IsSupported(CpuIsaLevel.Avx512) && AvxVnni.V512.IsSupported,
        _ => false,
    };

    /// <summary>Every tier the host supports, lowest first.</summary>
    public static IReadOnlyList<CpuIsaLevel> Supported { get; } = [.. Enum.GetValues<CpuIsaLevel>().Where(IsSupported)];

    /// <summary>Lower-case name of a tier as accepted by <c>SHARPLIGHT_CPU_ISA</c>.</summary>
    public static string Name(CpuIsaLevel level) => level switch
    {
        CpuIsaLevel.Scalar => "scalar",
        CpuIsaLevel.Vector128 => "v128",
        CpuIsaLevel.Neon => "neon",
        CpuIsaLevel.NeonDot => "neondot",
        CpuIsaLevel.Avx2 => "avx2",
        CpuIsaLevel.AvxVnni => "avxvnni",
        CpuIsaLevel.Avx512 => "avx512",
        _ => "avx512vnni",
    };

    private static CpuIsaLevel? Parse(string? text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return null;
        }

        foreach (var level in Enum.GetValues<CpuIsaLevel>())
        {
            if (string.Equals(Name(level), text.Trim(), StringComparison.OrdinalIgnoreCase))
            {
                return level;
            }
        }

        throw new InvalidOperationException($"{EnvironmentVariable}='{text}' is not one of scalar, v128, neon, neondot, avx2, avxvnni, avx512, avx512vnni.");
    }

    private static CpuIsaLevel Highest(CpuIsaLevel cap)
    {
        var level = cap;
        while (level > CpuIsaLevel.Scalar && !IsSupported(level))
        {
            level--;
        }

        return level;
    }
}
