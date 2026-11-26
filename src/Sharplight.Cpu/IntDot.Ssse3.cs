using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Sharplight.Cpu;

// The 128-bit x86 tier of the integer dot (v128 with SSSE3, an x86-64-v2 host or SHARPLIGHT_CPU_ISA=v128): pmaddubsw +
// pmaddwd, the 128-bit form of the AVX2 vpmaddubsw kernel. Codes of 128 and up take the same sign trick (|w| * sign(q, w) with
// w = u - 128), whose 128 * a * sum(q) is added back with the offsets. Float multiply-adds go through Vec128.MultiplyAdd (FMA3
// when present, else mul + add).
internal static unsafe partial class IntDot
{
    /// <summary>Whether the 128-bit x86 tier has the SSSE3 kernel (<c>pmaddubsw</c>); otherwise it widens to 16 bits.</summary>
    public static bool HasSsse3Dot => Ssse3.IsSupported;

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static float DotSsse3(byte* u, float* a, float* b, bool smallCodes, Q8Row x, int k)
    {
        if (!Ssse3.IsSupported)
        {
            throw new PlatformNotSupportedException();
        }

        var acc0 = Vector128<float>.Zero;
        var acc1 = Vector128<float>.Zero;
        var ones = Vector128.Create((short)1);
        var flip = Vector128.Create((byte)0x80);
        int i = 0;
        for (; i + 2 * GroupSize <= k; i += 2 * GroupSize)
        {
            acc0 = StepSsse3(u + i, a[i / GroupSize], x.Q + i, x.LaneScales + i / 4, smallCodes, ones, flip, acc0);
            acc1 = StepSsse3(u + i + GroupSize, a[i / GroupSize + 1], x.Q + i + GroupSize, x.LaneScales + i / 4 + 4, smallCodes, ones, flip, acc1);
        }

        if (i < k)
        {
            acc0 = StepSsse3(u + i, a[i / GroupSize], x.Q + i, x.LaneScales + i / 4, smallCodes, ones, flip, acc0);
        }

        return Vector128.Sum(acc0 + acc1) + Offsets128(a, b, smallCodes, x, k);
    }

    /// <summary>One 16-value group: four int32 lanes of u * q (pmaddubsw, pmaddwd), scaled by a * dx into the float accumulator.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> StepSsse3(byte* u, float a, sbyte* q, float* laneScales, bool smallCodes, Vector128<short> ones,
        Vector128<byte> flip, Vector128<float> acc)
    {
        var uv = Vector128.Load(u);
        var qv = Vector128.Load(q);
        Vector128<short> pairs;
        if (smallCodes)
        {
            pairs = Ssse3.MultiplyAddAdjacent(uv, qv);
        }
        else
        {
            var w = (uv ^ flip).AsSByte();
            pairs = Ssse3.MultiplyAddAdjacent(Ssse3.Abs(w), Ssse3.Sign(qv, w));
        }

        var lanes = Sse2.MultiplyAddAdjacent(pairs, ones);
        var scale = Vector128.Load(laneScales) * Vector128.Create(a);
        return Vec128.MultiplyAdd(Vector128.ConvertToSingle(lanes), scale, acc);
    }

    /// <summary>sum((b + shift * a) * groupSum) over the row, shift = 128 with the sign trick (else 0).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static float Offsets128(float* a, float* b, bool smallCodes, Q8Row x, int k)
    {
        int groups = k / GroupSize;
        var acc0 = Vector128<float>.Zero;
        var acc1 = Vector128<float>.Zero;
        float shiftScalar = smallCodes ? 0f : 128f;
        var shift = Vector128.Create(shiftScalar);
        int g = 0;
        for (; g + 8 <= groups; g += 8)
        {
            var b0 = Vec128.MultiplyAdd(Vector128.Load(a + g), shift, Vector128.Load(b + g));
            var b1 = Vec128.MultiplyAdd(Vector128.Load(a + g + 4), shift, Vector128.Load(b + g + 4));
            acc0 = Vec128.MultiplyAdd(b0, Vector128.Load(x.GroupSums + g), acc0);
            acc1 = Vec128.MultiplyAdd(b1, Vector128.Load(x.GroupSums + g + 4), acc1);
        }

        float s = Vector128.Sum(acc0 + acc1);
        for (; g < groups; g++)
        {
            s += (b[g] + shiftScalar * a[g]) * x.GroupSums[g];
        }

        return s;
    }
}
