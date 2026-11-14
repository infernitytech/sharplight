using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;

namespace Sharplight.Cpu;

/// <summary>
/// Eight float lanes as a pair of 128-bit vectors (Arm64 NEON): the lane count of the AVX2 tier, so every broadcast feeds two
/// fused multiply-adds and the tiled kernels keep their task shapes.
/// </summary>
internal readonly unsafe struct FloatLanes128x2(Vector128<float> lower, Vector128<float> upper) : IFloatLanes<FloatLanes128x2>
{
    private readonly Vector128<float> _lower = lower;
    private readonly Vector128<float> _upper = upper;

    public static int Count => 8;

    public static FloatLanes128x2 Zero => new(Vector128<float>.Zero, Vector128<float>.Zero);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static FloatLanes128x2 Broadcast(float value)
    {
        var v = Vector128.Create(value);
        return new(v, v);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static FloatLanes128x2 Load(float* p) => new(Vector128.Load(p), Vector128.Load(p + 4));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Store(FloatLanes128x2 v, float* p)
    {
        v._lower.Store(p);
        v._upper.Store(p + 4);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static FloatLanes128x2 Fma(FloatLanes128x2 a, FloatLanes128x2 b, FloatLanes128x2 c) =>
        new(AdvSimd.FusedMultiplyAdd(c._lower, a._lower, b._lower), AdvSimd.FusedMultiplyAdd(c._upper, a._upper, b._upper));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static FloatLanes128x2 Multiply(FloatLanes128x2 a, FloatLanes128x2 b) => new(a._lower * b._lower, a._upper * b._upper);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static FloatLanes128x2 Add(FloatLanes128x2 a, FloatLanes128x2 b) => new(a._lower + b._lower, a._upper + b._upper);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static FloatLanes128x2 Subtract(FloatLanes128x2 a, FloatLanes128x2 b) => new(a._lower - b._lower, a._upper - b._upper);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static FloatLanes128x2 Max(FloatLanes128x2 a, FloatLanes128x2 b) =>
        new(Vector128.Max(a._lower, b._lower), Vector128.Max(a._upper, b._upper));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static FloatLanes128x2 Exp(FloatLanes128x2 x) => new(Vector128.Exp(x._lower), Vector128.Exp(x._upper));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static FloatLanes128x2 ZeroIfNegativeInfinity(FloatLanes128x2 v)
    {
        var negativeInfinity = Vector128.Create(float.NegativeInfinity);
        return new(
            Vector128.ConditionalSelect(Vector128.Equals(v._lower, negativeInfinity), Vector128<float>.Zero, v._lower),
            Vector128.ConditionalSelect(Vector128.Equals(v._upper, negativeInfinity), Vector128<float>.Zero, v._upper));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static FloatLanes128x2 KeepInRange(FloatLanes128x2 v, int j, int* first, int* end)
    {
        var jv = Vector128.Create(j);
        var negativeInfinity = Vector128.Create(float.NegativeInfinity);
        var lower = Vector128.GreaterThanOrEqual(jv, Vector128.Load(first)) & Vector128.LessThan(jv, Vector128.Load(end));
        var upper = Vector128.GreaterThanOrEqual(jv, Vector128.Load(first + 4)) & Vector128.LessThan(jv, Vector128.Load(end + 4));
        return new(
            Vector128.ConditionalSelect(lower.AsSingle(), v._lower, negativeInfinity),
            Vector128.ConditionalSelect(upper.AsSingle(), v._upper, negativeInfinity));
    }
}
