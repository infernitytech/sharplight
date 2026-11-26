using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace Sharplight.Cpu;

/// <summary>Portable 128-bit helpers of the v128 and NEON kernels.</summary>
internal static class Vec128
{
    /// <summary>
    /// a * b + c: one fused instruction with x86 FMA3 or Arm64 NEON, else a separate multiply and add (an SSE-only x86 host;
    /// <see cref="Vector128.FusedMultiplyAdd(Vector128{float}, Vector128{float}, Vector128{float})"/> would fall back to a per-lane
    /// software fma there, which is far slower than the unfused pair).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> MultiplyAdd(Vector128<float> a, Vector128<float> b, Vector128<float> c)
    {
        if (Fma.IsSupported)
        {
            return Fma.MultiplyAdd(a, b, c);
        }

        if (AdvSimd.IsSupported)
        {
            return AdvSimd.FusedMultiplyAdd(c, a, b);
        }

        return (a * b) + c;
    }
}
