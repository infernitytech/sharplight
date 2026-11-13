using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Sharplight.Cpu;

// The per-tier dot kernels. Each sums a * dx * sum(u * q) over int32 lanes of four values and adds the offset term
// sum(b * groupSum) separately. The 8-bit sign trick (|w| * sign(q, w) with w = u - 128) keeps vpmaddubsw exact for codes up to 255.
internal static unsafe partial class IntDot
{
    private static float DotScalar(byte* u, float* a, float* b, Q8Row x, int k)
    {
        float acc = 0f;
        for (int g = 0; g < k / GroupSize; g++)
        {
            int s = 0;
            byte* ug = u + g * GroupSize;
            sbyte* qg = x.Q + g * GroupSize;
            for (int i = 0; i < GroupSize; i++)
            {
                s += ug[i] * qg[i];
            }

            acc += a[g] * x.LaneScales[g * 4] * s + b[g] * x.GroupSums[g];
        }

        return acc;
    }

    private static float DotVector128(byte* u, float* a, float* b, Q8Row x, int k)
    {
        float acc = 0f;
        for (int g = 0; g < k / GroupSize; g++)
        {
            var (ulo, uhi) = Vector128.Widen(Vector128.Load(u + g * GroupSize));
            var (qlo, qhi) = Vector128.Widen(Vector128.Load(x.Q + g * GroupSize));
            var (p0, p1) = Vector128.Widen(ulo.AsInt16() * qlo);
            var (p2, p3) = Vector128.Widen(uhi.AsInt16() * qhi);
            int s = Vector128.Sum((p0 + p1) + (p2 + p3));
            acc += a[g] * x.LaneScales[g * 4] * s + b[g] * x.GroupSums[g];
        }

        return acc;
    }

    private static float DotAvx2(byte* u, float* a, float* b, bool smallCodes, Q8Row x, int k)
    {
        if (!Avx2.IsSupported || !Fma.IsSupported)
        {
            throw new PlatformNotSupportedException();
        }

        var acc0 = Vector256<float>.Zero;
        var acc1 = Vector256<float>.Zero;
        var ones = Vector256.Create((short)1);
        var flip = Vector256.Create((byte)0x80);
        int i = 0;
        for (; i + 64 <= k; i += 64)
        {
            acc0 = Step256(u + i, a + i / GroupSize, x.Q + i, x.LaneScales + i / 4, smallCodes, ones, flip, acc0);
            acc1 = Step256(u + i + 32, a + i / GroupSize + 2, x.Q + i + 32, x.LaneScales + i / 4 + 8, smallCodes, ones, flip, acc1);
        }

        if (i < k)
        {
            acc0 = Step256(u + i, a + i / GroupSize, x.Q + i, x.LaneScales + i / 4, smallCodes, ones, flip, acc0);
        }

        return Vector256.Sum(acc0 + acc1) + Offsets256(a, b, smallCodes, x, k);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> Step256(byte* u, float* a, sbyte* q, float* laneScales, bool smallCodes, Vector256<short> ones,
        Vector256<byte> flip, Vector256<float> acc)
    {
        var uv = Vector256.Load(u);
        var qv = Vector256.Load(q);
        Vector256<short> pairs;
        if (smallCodes)
        {
            pairs = Avx2.MultiplyAddAdjacent(uv, qv);
        }
        else
        {
            var w = (uv ^ flip).AsSByte();
            pairs = Avx2.MultiplyAddAdjacent(Avx2.Abs(w), Avx2.Sign(qv, w));
        }

        var lanes = Avx2.MultiplyAddAdjacent(pairs, ones);
        var scale = Vector256.Create(Vector128.Create(a[0]), Vector128.Create(a[1])) * Vector256.Load(laneScales);
        return Vector256.FusedMultiplyAdd(Vector256.ConvertToSingle(lanes), scale, acc);
    }

    /// <summary>
    /// sum(b * groupSum) over the row; with the sign trick the codes were used as u - 128, which moves 128 * a into the offset.
    /// </summary>
    private static float Offsets256(float* a, float* b, bool smallCodes, Q8Row x, int k)
    {
        int groups = k / GroupSize;
        var acc = Vector256<float>.Zero;
        var shift = Vector256.Create(smallCodes ? 0f : 128f);
        int g = 0;
        for (; g + 8 <= groups; g += 8)
        {
            var bv = Vector256.FusedMultiplyAdd(Vector256.Load(a + g), shift, Vector256.Load(b + g));
            acc = Vector256.FusedMultiplyAdd(bv, Vector256.Load(x.GroupSums + g), acc);
        }

        float s = Vector256.Sum(acc);
        float shiftScalar = smallCodes ? 0f : 128f;
        for (; g < groups; g++)
        {
            s += (b[g] + shiftScalar * a[g]) * x.GroupSums[g];
        }

        return s;
    }

    private static float DotAvxVnni(byte* u, float* a, float* b, Q8Row x, int k)
    {
        if (!AvxVnni.IsSupported || !Fma.IsSupported)
        {
            throw new PlatformNotSupportedException();
        }

        var acc0 = Vector256<float>.Zero;
        var acc1 = Vector256<float>.Zero;
        int i = 0;
        for (; i + 64 <= k; i += 64)
        {
            acc0 = StepVnni256(u + i, a + i / GroupSize, x.Q + i, x.LaneScales + i / 4, acc0);
            acc1 = StepVnni256(u + i + 32, a + i / GroupSize + 2, x.Q + i + 32, x.LaneScales + i / 4 + 8, acc1);
        }

        if (i < k)
        {
            acc0 = StepVnni256(u + i, a + i / GroupSize, x.Q + i, x.LaneScales + i / 4, acc0);
        }

        return Vector256.Sum(acc0 + acc1) + Offsets256(a, b, smallCodes: true, x, k);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> StepVnni256(byte* u, float* a, sbyte* q, float* laneScales, Vector256<float> acc)
    {
        var lanes = AvxVnni.MultiplyWideningAndAdd(Vector256<int>.Zero, Vector256.Load(u), Vector256.Load(q));
        var scale = Vector256.Create(Vector128.Create(a[0]), Vector128.Create(a[1])) * Vector256.Load(laneScales);
        return Vector256.FusedMultiplyAdd(Vector256.ConvertToSingle(lanes), scale, acc);
    }

    private static float DotAvx512(byte* u, float* a, float* b, bool smallCodes, Q8Row x, int k)
    {
        if (!Avx512BW.IsSupported || !Avx512F.IsSupported || !Avx2.IsSupported || !Fma.IsSupported)
        {
            throw new PlatformNotSupportedException();
        }

        var acc = Vector512<float>.Zero;
        var ones = Vector512.Create((short)1);
        var flip = Vector512.Create((byte)0x80);
        var spread = Vector512.Create(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);
        int i = 0;
        for (; i + 64 <= k; i += 64)
        {
            var uv = Vector512.Load(u + i);
            var qv = Vector512.Load(x.Q + i);
            Vector512<short> pairs;
            if (smallCodes)
            {
                pairs = Avx512BW.MultiplyAddAdjacent(uv, qv);
            }
            else
            {
                var w = (uv ^ flip).AsSByte();
                var signed = Vector512.ConditionalSelect(Vector512.LessThan(w, Vector512<sbyte>.Zero), -qv, qv);
                pairs = Avx512BW.MultiplyAddAdjacent(Avx512BW.Abs(w), signed);
            }

            var lanes = Avx512BW.MultiplyAddAdjacent(pairs, ones);
            var scale = Spread4(a + i / GroupSize, spread) * Vector512.Load(x.LaneScales + i / 4);
            acc = Vector512.FusedMultiplyAdd(Vector512.ConvertToSingle(lanes), scale, acc);
        }

        var tail = Vector256<float>.Zero;
        if (i < k)
        {
            tail = Step256(u + i, a + i / GroupSize, x.Q + i, x.LaneScales + i / 4, smallCodes, Vector256.Create((short)1), Vector256.Create((byte)0x80), tail);
        }

        return Vector512.Sum(acc) + Vector256.Sum(tail) + Offsets512(a, b, smallCodes, x, k);
    }

    private static float DotAvx512Vnni(byte* u, float* a, float* b, Q8Row x, int k)
    {
        if (!AvxVnni.V512.IsSupported || !Avx512F.IsSupported)
        {
            throw new PlatformNotSupportedException();
        }

        var acc0 = Vector512<float>.Zero;
        var acc1 = Vector512<float>.Zero;
        var spread = Vector512.Create(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);
        int i = 0;
        for (; i + 128 <= k; i += 128)
        {
            acc0 = StepVnni512(u + i, a + i / GroupSize, x.Q + i, x.LaneScales + i / 4, spread, acc0);
            acc1 = StepVnni512(u + i + 64, a + i / GroupSize + 4, x.Q + i + 64, x.LaneScales + i / 4 + 16, spread, acc1);
        }

        if (i + 64 <= k)
        {
            acc0 = StepVnni512(u + i, a + i / GroupSize, x.Q + i, x.LaneScales + i / 4, spread, acc0);
            i += 64;
        }

        if (i < k)
        {
            // 32 values left: the upper halves are zero, so their lanes add nothing.
            var lanes = AvxVnni.V512.MultiplyWideningAndAdd(Vector512<int>.Zero, Vector256.Load(u + i).ToVector512(), Vector256.Load(x.Q + i).ToVector512());
            var scale = (Vector256.Create(Vector128.Create(a[i / GroupSize]), Vector128.Create(a[i / GroupSize + 1])) * Vector256.Load(x.LaneScales + i / 4)).ToVector512();
            acc1 = Vector512.FusedMultiplyAdd(Vector512.ConvertToSingle(lanes), scale, acc1);
        }

        return Vector512.Sum(acc0 + acc1) + Offsets512(a, b, smallCodes: true, x, k);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<float> StepVnni512(byte* u, float* a, sbyte* q, float* laneScales, Vector512<int> spread, Vector512<float> acc)
    {
        var lanes = AvxVnni.V512.MultiplyWideningAndAdd(Vector512<int>.Zero, Vector512.Load(u), Vector512.Load(q));
        var scale = Spread4(a, spread) * Vector512.Load(laneScales);
        return Vector512.FusedMultiplyAdd(Vector512.ConvertToSingle(lanes), scale, acc);
    }

    /// <summary>Four multipliers (64 values) repeated over the four int32 lanes of their 16 values.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<float> Spread4(float* a, Vector512<int> spread) =>
        Avx512F.PermuteVar16x32(Vector128.Load(a).ToVector256Unsafe().ToVector512Unsafe(), spread);

    private static float Offsets512(float* a, float* b, bool smallCodes, Q8Row x, int k)
    {
        int groups = k / GroupSize;
        var acc = Vector512<float>.Zero;
        var shift = Vector512.Create(smallCodes ? 0f : 128f);
        int g = 0;
        for (; g + 16 <= groups; g += 16)
        {
            var bv = Vector512.FusedMultiplyAdd(Vector512.Load(a + g), shift, Vector512.Load(b + g));
            acc = Vector512.FusedMultiplyAdd(bv, Vector512.Load(x.GroupSums + g), acc);
        }

        float s = Vector512.Sum(acc);
        float shiftScalar = smallCodes ? 0f : 128f;
        for (; g < groups; g++)
        {
            s += (b[g] + shiftScalar * a[g]) * x.GroupSums[g];
        }

        return s;
    }
}
