using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;

namespace Sharplight.Cpu;

// The Arm64 tiers of the integer dot: activation quantization with NEON (bit-identical to the scalar reference) and the
// sdot kernel. sdot multiplies signed bytes only, so the codes enter as w = u - 128 (exact for every code 0..255) and the
// 128 * a[g] * sum(q) the shift removes is added back with the offsets, as in the x86 sign-trick path.
internal static unsafe partial class IntDot
{
    private static void QuantizeBlockNeon(float* x, sbyte* q, float* laneScales, float* groupSums)
    {
        var v0 = Vector128.Load(x);
        var v1 = Vector128.Load(x + 4);
        var v2 = Vector128.Load(x + 8);
        var v3 = Vector128.Load(x + 12);
        var v4 = Vector128.Load(x + 16);
        var v5 = Vector128.Load(x + 20);
        var v6 = Vector128.Load(x + 24);
        var v7 = Vector128.Load(x + 28);
        var m = Vector128.Max(
            Vector128.Max(Vector128.Max(Vector128.Abs(v0), Vector128.Abs(v1)), Vector128.Max(Vector128.Abs(v2), Vector128.Abs(v3))),
            Vector128.Max(Vector128.Max(Vector128.Abs(v4), Vector128.Abs(v5)), Vector128.Max(Vector128.Abs(v6), Vector128.Abs(v7))));
        float amax = AdvSimd.Arm64.MaxAcross(m).ToScalar();
        float d = amax / 127f;
        float id = d != 0f ? 1f / d : 0f;
        var idv = Vector128.Create(id);
        var half = Vector128.Create(0.5f);
        var i0 = Round(v0 * idv, half);
        var i1 = Round(v1 * idv, half);
        var i2 = Round(v2 * idv, half);
        var i3 = Round(v3 * idv, half);
        var i4 = Round(v4 * idv, half);
        var i5 = Round(v5 * idv, half);
        var i6 = Round(v6 * idv, half);
        var i7 = Round(v7 * idv, half);
        Vector128.Narrow(Vector128.Narrow(i0, i1), Vector128.Narrow(i2, i3)).Store(q);
        Vector128.Narrow(Vector128.Narrow(i4, i5), Vector128.Narrow(i6, i7)).Store(q + 16);
        var dv = Vector128.Create(d);
        dv.Store(laneScales);
        dv.Store(laneScales + 4);
        groupSums[0] = d * Vector128.Sum((i0 + i1) + (i2 + i3));
        groupSums[1] = d * Vector128.Sum((i4 + i5) + (i6 + i7));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<int> Round(Vector128<float> v, Vector128<float> half) =>
        Vector128.ConvertToInt32(v + Vector128.CopySign(half, v));

    private static float DotNeonDot(byte* u, float* a, float* b, Q8Row x, int k)
    {
        if (!Dp.IsSupported || !AdvSimd.Arm64.IsSupported)
        {
            throw new PlatformNotSupportedException();
        }

        var acc0 = Vector128<float>.Zero;
        var acc1 = Vector128<float>.Zero;
        var flip = Vector128.Create((byte)0x80);
        int i = 0;
        for (; i + 2 * GroupSize <= k; i += 2 * GroupSize)
        {
            acc0 = StepNeonDot(u + i, a[i / GroupSize], x.Q + i, x.LaneScales + i / 4, flip, acc0);
            acc1 = StepNeonDot(u + i + GroupSize, a[i / GroupSize + 1], x.Q + i + GroupSize, x.LaneScales + i / 4 + 4, flip, acc1);
        }

        if (i < k)
        {
            acc0 = StepNeonDot(u + i, a[i / GroupSize], x.Q + i, x.LaneScales + i / 4, flip, acc0);
        }

        return Vector128.Sum(acc0 + acc1) + OffsetsNeon(a, b, x, k);
    }

    /// <summary>One 16-value group: four int32 lanes of (u - 128) * q, scaled by a * dx into the float accumulator.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> StepNeonDot(byte* u, float a, sbyte* q, float* laneScales, Vector128<byte> flip, Vector128<float> acc)
    {
        var w = (Vector128.Load(u) ^ flip).AsSByte();
        var lanes = Dp.DotProduct(Vector128<int>.Zero, w, Vector128.Load(q));
        var scale = Vector128.Load(laneScales) * Vector128.Create(a);
        return AdvSimd.FusedMultiplyAdd(acc, AdvSimd.ConvertToSingle(lanes), scale);
    }

    /// <summary>sum((b + 128 * a) * groupSum) over the row: the offsets plus the term the u - 128 shift removed.</summary>
    private static float OffsetsNeon(float* a, float* b, Q8Row x, int k)
    {
        int groups = k / GroupSize;
        var acc = Vector128<float>.Zero;
        var shift = Vector128.Create(128f);
        int g = 0;
        for (; g + 4 <= groups; g += 4)
        {
            var bv = AdvSimd.FusedMultiplyAdd(Vector128.Load(b + g), Vector128.Load(a + g), shift);
            acc = AdvSimd.FusedMultiplyAdd(acc, bv, Vector128.Load(x.GroupSums + g));
        }

        float s = Vector128.Sum(acc);
        for (; g < groups; g++)
        {
            s += (b[g] + 128f * a[g]) * x.GroupSums[g];
        }

        return s;
    }
}
