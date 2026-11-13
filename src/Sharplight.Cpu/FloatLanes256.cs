using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace Sharplight.Cpu;

/// <summary>Eight float lanes (AVX2).</summary>
internal readonly unsafe struct FloatLanes256(Vector256<float> value) : IFloatLanes<FloatLanes256>
{
    private readonly Vector256<float> _value = value;

    public static int Count => 8;

    public static FloatLanes256 Zero => new(Vector256<float>.Zero);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static FloatLanes256 Broadcast(float value) => new(Vector256.Create(value));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static FloatLanes256 Load(float* p) => new(Vector256.Load(p));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Store(FloatLanes256 v, float* p) => v._value.Store(p);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static FloatLanes256 Fma(FloatLanes256 a, FloatLanes256 b, FloatLanes256 c) => new(Vector256.FusedMultiplyAdd(a._value, b._value, c._value));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static FloatLanes256 Multiply(FloatLanes256 a, FloatLanes256 b) => new(a._value * b._value);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static FloatLanes256 Add(FloatLanes256 a, FloatLanes256 b) => new(a._value + b._value);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static FloatLanes256 Subtract(FloatLanes256 a, FloatLanes256 b) => new(a._value - b._value);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static FloatLanes256 Max(FloatLanes256 a, FloatLanes256 b) => new(Vector256.Max(a._value, b._value));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static FloatLanes256 Exp(FloatLanes256 x) => new(Vector256.Exp(x._value));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static FloatLanes256 ZeroIfNegativeInfinity(FloatLanes256 v) =>
        new(Vector256.ConditionalSelect(Vector256.Equals(v._value, Vector256.Create(float.NegativeInfinity)), Vector256<float>.Zero, v._value));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static FloatLanes256 KeepInRange(FloatLanes256 v, int j, int* first, int* end)
    {
        var jv = Vector256.Create(j);
        var inside = Vector256.GreaterThanOrEqual(jv, Vector256.Load(first)) & Vector256.LessThan(jv, Vector256.Load(end));
        return new(Vector256.ConditionalSelect(inside.AsSingle(), v._value, Vector256.Create(float.NegativeInfinity)));
    }
}
