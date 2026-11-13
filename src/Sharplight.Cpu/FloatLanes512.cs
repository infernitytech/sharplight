using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace Sharplight.Cpu;

/// <summary>Sixteen float lanes (AVX-512).</summary>
internal readonly unsafe struct FloatLanes512(Vector512<float> value) : IFloatLanes<FloatLanes512>
{
    private readonly Vector512<float> _value = value;

    public static int Count => 16;

    public static FloatLanes512 Zero => new(Vector512<float>.Zero);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static FloatLanes512 Broadcast(float value) => new(Vector512.Create(value));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static FloatLanes512 Load(float* p) => new(Vector512.Load(p));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Store(FloatLanes512 v, float* p) => v._value.Store(p);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static FloatLanes512 Fma(FloatLanes512 a, FloatLanes512 b, FloatLanes512 c) => new(Vector512.FusedMultiplyAdd(a._value, b._value, c._value));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static FloatLanes512 Multiply(FloatLanes512 a, FloatLanes512 b) => new(a._value * b._value);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static FloatLanes512 Add(FloatLanes512 a, FloatLanes512 b) => new(a._value + b._value);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static FloatLanes512 Subtract(FloatLanes512 a, FloatLanes512 b) => new(a._value - b._value);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static FloatLanes512 Max(FloatLanes512 a, FloatLanes512 b) => new(Vector512.Max(a._value, b._value));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static FloatLanes512 Exp(FloatLanes512 x) => new(Vector512.Exp(x._value));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static FloatLanes512 ZeroIfNegativeInfinity(FloatLanes512 v) =>
        new(Vector512.ConditionalSelect(Vector512.Equals(v._value, Vector512.Create(float.NegativeInfinity)), Vector512<float>.Zero, v._value));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static FloatLanes512 KeepInRange(FloatLanes512 v, int j, int* first, int* end)
    {
        var jv = Vector512.Create(j);
        var inside = Vector512.GreaterThanOrEqual(jv, Vector512.Load(first)) & Vector512.LessThan(jv, Vector512.Load(end));
        return new(Vector512.ConditionalSelect(inside.AsSingle(), v._value, Vector512.Create(float.NegativeInfinity)));
    }
}
