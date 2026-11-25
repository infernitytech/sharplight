using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace Sharplight.Cpu;

/// <summary>
/// tanh, softplus and mish of float lanes on <see cref="VectorExp"/> (the runtime's scalar <c>MathF.Tanh</c> and the double-precision
/// <c>log(1 + exp(x))</c> took 12, 10 and 31 ns per value): tanh is Cephes' <c>tanhf</c> (an odd polynomial below |x| = 0.625, else
/// <c>1 - 2 / (e^2|x| + 1)</c>), softplus is <c>max(x, 0) + log1p(e^-|x|)</c> with log1p from Cephes' <c>logf</c> and Goldberg's
/// correction, and mish is <c>x * n / (n + 2)</c> with <c>n = e^x (e^x + 2)</c> (<c>tanh(log(1 + e^x))</c> written out). All of them stay
/// within a few float ulp of the double-precision results; softplus and mish return x above 20, as llama.cpp's softplus does.
/// </summary>
internal static class VectorActivations
{
    private const float TanhPolynomialBelow = 0.625f;
    private const float T0 = -5.70498872745e-3f;
    private const float T1 = 2.06390887954e-2f;
    private const float T2 = -5.37397155531e-2f;
    private const float T3 = 1.33314422036e-1f;
    private const float T4 = -3.33332819422e-1f;
    private const float LinearAbove = 20f;
    private const float SqrtHalf = 0.707106781186547524f;
    private const float L0 = 7.0376836292e-2f;
    private const float L1 = -1.1514610310e-1f;
    private const float L2 = 1.1676998740e-1f;
    private const float L3 = -1.2420140846e-1f;
    private const float L4 = 1.4249322787e-1f;
    private const float L5 = -1.6668057665e-1f;
    private const float L6 = 2.0000714765e-1f;
    private const float L7 = -2.4999993993e-1f;
    private const float L8 = 3.3333331174e-1f;
    private const float Ln2High = 0.693359375f;
    private const float Ln2Low = -2.12194440e-4f;

    /// <summary>tanh of eight lanes.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Tanh(Vector256<float> x)
    {
        var a = Vector256.Abs(x);
        var z = x * x;
        var p = Vector256.FusedMultiplyAdd(Vector256.Create(T0), z, Vector256.Create(T1));
        p = Vector256.FusedMultiplyAdd(p, z, Vector256.Create(T2));
        p = Vector256.FusedMultiplyAdd(p, z, Vector256.Create(T3));
        p = Vector256.FusedMultiplyAdd(p, z, Vector256.Create(T4));
        var small = Vector256.FusedMultiplyAdd(p * z, a, a);
        var large = Vector256<float>.One - Vector256.Create(2f) / (VectorExp.Exp(a + a) + Vector256<float>.One);
        var magnitude = Vector256.ConditionalSelect(Vector256.LessThan(a, Vector256.Create(TanhPolynomialBelow)), small, large);
        return Vector256.ConditionalSelect(Vector256.Create(-0f), x, magnitude);
    }

    /// <summary>tanh of four lanes.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> Tanh(Vector128<float> x)
    {
        var a = Vector128.Abs(x);
        var z = x * x;
        var p = Vector128.FusedMultiplyAdd(Vector128.Create(T0), z, Vector128.Create(T1));
        p = Vector128.FusedMultiplyAdd(p, z, Vector128.Create(T2));
        p = Vector128.FusedMultiplyAdd(p, z, Vector128.Create(T3));
        p = Vector128.FusedMultiplyAdd(p, z, Vector128.Create(T4));
        var small = Vector128.FusedMultiplyAdd(p * z, a, a);
        var large = Vector128<float>.One - Vector128.Create(2f) / (VectorExp.Exp(a + a) + Vector128<float>.One);
        var magnitude = Vector128.ConditionalSelect(Vector128.LessThan(a, Vector128.Create(TanhPolynomialBelow)), small, large);
        return Vector128.ConditionalSelect(Vector128.Create(-0f), x, magnitude);
    }

    /// <summary>softplus of eight lanes: <c>log(1 + e^x)</c>, x itself above 20.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Softplus(Vector256<float> x)
    {
        var result = Vector256.MaxNative(x, Vector256<float>.Zero) + Log1p(VectorExp.Exp(-Vector256.Abs(x)));
        return Vector256.ConditionalSelect(Vector256.GreaterThan(x, Vector256.Create(LinearAbove)), x, result);
    }

    /// <summary>softplus of four lanes: <c>log(1 + e^x)</c>, x itself above 20.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> Softplus(Vector128<float> x)
    {
        var result = Vector128.MaxNative(x, Vector128<float>.Zero) + Log1p(VectorExp.Exp(-Vector128.Abs(x)));
        return Vector128.ConditionalSelect(Vector128.GreaterThan(x, Vector128.Create(LinearAbove)), x, result);
    }

    /// <summary>mish of eight lanes: <c>x tanh(softplus(x))</c>, x itself above 20.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Mish(Vector256<float> x)
    {
        var e = VectorExp.Exp(Vector256.MinNative(x, Vector256.Create(LinearAbove)));
        var n = e * (e + Vector256.Create(2f));
        var result = x * n / (n + Vector256.Create(2f));
        return Vector256.ConditionalSelect(Vector256.GreaterThan(x, Vector256.Create(LinearAbove)), x, result);
    }

    /// <summary>mish of four lanes: <c>x tanh(softplus(x))</c>, x itself above 20.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> Mish(Vector128<float> x)
    {
        var e = VectorExp.Exp(Vector128.MinNative(x, Vector128.Create(LinearAbove)));
        var n = e * (e + Vector128.Create(2f));
        var result = x * n / (n + Vector128.Create(2f));
        return Vector128.ConditionalSelect(Vector128.GreaterThan(x, Vector128.Create(LinearAbove)), x, result);
    }

    /// <summary>
    /// log(1 + u) for u in [0, 1]: <c>log(w) * u / (w - 1)</c> with <c>w = 1 + u</c> rounded (Goldberg), which cancels the rounding of w;
    /// u itself where w rounds to 1.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> Log1p(Vector256<float> u)
    {
        var w = u + Vector256<float>.One;
        var d = w - Vector256<float>.One;
        var corrected = Log(w) * (u / d);
        return Vector256.ConditionalSelect(Vector256.Equals(d, Vector256<float>.Zero), u, corrected);
    }

    /// <summary>log(1 + u) for u in [0, 1] (see the eight-lane version).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> Log1p(Vector128<float> u)
    {
        var w = u + Vector128<float>.One;
        var d = w - Vector128<float>.One;
        var corrected = Log(w) * (u / d);
        return Vector128.ConditionalSelect(Vector128.Equals(d, Vector128<float>.Zero), u, corrected);
    }

    /// <summary>Natural log of positive normal lanes (Cephes <c>logf</c>: the mantissa in [√½, √2) minus 1 through a degree-9 polynomial).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> Log(Vector256<float> x)
    {
        var bits = x.AsInt32();
        var e = Vector256.ConvertToSingle((bits >> 23) - Vector256.Create(126));
        var m = ((bits & Vector256.Create(0x007FFFFF)) | Vector256.Create(0x3F000000)).AsSingle();   // in [0.5, 1)
        var below = Vector256.LessThan(m, Vector256.Create(SqrtHalf));
        e -= Vector256.ConditionalSelect(below, Vector256<float>.One, Vector256<float>.Zero);
        var r = m + Vector256.ConditionalSelect(below, m, Vector256<float>.Zero) - Vector256<float>.One;
        var z = r * r;
        var p = Vector256.FusedMultiplyAdd(Vector256.Create(L0), r, Vector256.Create(L1));
        p = Vector256.FusedMultiplyAdd(p, r, Vector256.Create(L2));
        p = Vector256.FusedMultiplyAdd(p, r, Vector256.Create(L3));
        p = Vector256.FusedMultiplyAdd(p, r, Vector256.Create(L4));
        p = Vector256.FusedMultiplyAdd(p, r, Vector256.Create(L5));
        p = Vector256.FusedMultiplyAdd(p, r, Vector256.Create(L6));
        p = Vector256.FusedMultiplyAdd(p, r, Vector256.Create(L7));
        p = Vector256.FusedMultiplyAdd(p, r, Vector256.Create(L8));
        var y = Vector256.FusedMultiplyAdd(e, Vector256.Create(Ln2Low), p * r * z);
        y = Vector256.FusedMultiplyAdd(Vector256.Create(-0.5f), z, y);
        return Vector256.FusedMultiplyAdd(e, Vector256.Create(Ln2High), r + y);
    }

    /// <summary>Natural log of positive normal lanes (see the eight-lane version).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> Log(Vector128<float> x)
    {
        var bits = x.AsInt32();
        var e = Vector128.ConvertToSingle((bits >> 23) - Vector128.Create(126));
        var m = ((bits & Vector128.Create(0x007FFFFF)) | Vector128.Create(0x3F000000)).AsSingle();
        var below = Vector128.LessThan(m, Vector128.Create(SqrtHalf));
        e -= Vector128.ConditionalSelect(below, Vector128<float>.One, Vector128<float>.Zero);
        var r = m + Vector128.ConditionalSelect(below, m, Vector128<float>.Zero) - Vector128<float>.One;
        var z = r * r;
        var p = Vector128.FusedMultiplyAdd(Vector128.Create(L0), r, Vector128.Create(L1));
        p = Vector128.FusedMultiplyAdd(p, r, Vector128.Create(L2));
        p = Vector128.FusedMultiplyAdd(p, r, Vector128.Create(L3));
        p = Vector128.FusedMultiplyAdd(p, r, Vector128.Create(L4));
        p = Vector128.FusedMultiplyAdd(p, r, Vector128.Create(L5));
        p = Vector128.FusedMultiplyAdd(p, r, Vector128.Create(L6));
        p = Vector128.FusedMultiplyAdd(p, r, Vector128.Create(L7));
        p = Vector128.FusedMultiplyAdd(p, r, Vector128.Create(L8));
        var y = Vector128.FusedMultiplyAdd(e, Vector128.Create(Ln2Low), p * r * z);
        y = Vector128.FusedMultiplyAdd(Vector128.Create(-0.5f), z, y);
        return Vector128.FusedMultiplyAdd(e, Vector128.Create(Ln2High), r + y);
    }
}
