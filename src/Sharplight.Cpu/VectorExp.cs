using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace Sharplight.Cpu;

/// <summary>
/// e^x of float lanes: Cephes' <c>expf</c> (x = n ln 2 + r with |r| ≤ ln 2 / 2, a degree-6 polynomial for e^r, 2^n from the exponent bits),
/// within 2 ulp of the exact value over the normal range. The runtime's <c>Vector256.Exp</c> takes ~27 ns per value on AVX2 hosts (it runs
/// scalar), which made every SiLU, sigmoid and softmax of the CPU kernels slower than the products around them.
/// </summary>
/// <remarks>Results below e^-87.3 flush to zero; above e^88.72 they are +∞; NaN stays NaN.</remarks>
internal static class VectorExp
{
    private const float Log2E = 1.44269504088896341f;
    private const float Ln2High = 0.693359375f;
    private const float Ln2Low = -2.12194440e-4f;
    private const float Lowest = -87.3365448f;
    private const float Highest = 88.7228394f;
    private const float P0 = 1.9875691500e-4f;
    private const float P1 = 1.3981999507e-3f;
    private const float P2 = 8.3334519073e-3f;
    private const float P3 = 4.1665795894e-2f;
    private const float P4 = 1.6666665459e-1f;
    private const float P5 = 5.0000001201e-1f;

    /// <summary>e^x of eight lanes.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Exp(Vector256<float> x)
    {
        var clamped = Vector256.MinNative(Vector256.MaxNative(x, Vector256.Create(Lowest)), Vector256.Create(Highest));
        var n = Vector256.Round(clamped * Log2E);
        var r = Vector256.FusedMultiplyAdd(n, Vector256.Create(-Ln2Low), Vector256.FusedMultiplyAdd(n, Vector256.Create(-Ln2High), clamped));
        var p = Vector256.FusedMultiplyAdd(Vector256.Create(P0), r, Vector256.Create(P1));
        p = Vector256.FusedMultiplyAdd(p, r, Vector256.Create(P2));
        p = Vector256.FusedMultiplyAdd(p, r, Vector256.Create(P3));
        p = Vector256.FusedMultiplyAdd(p, r, Vector256.Create(P4));
        p = Vector256.FusedMultiplyAdd(p, r, Vector256.Create(P5));
        p = Vector256.FusedMultiplyAdd(p, r * r, r + Vector256<float>.One);
        // 2^n for n in [-126, 128]: n = 128 only at the top of the range, as two factors so the exponent stays in range.
        var exponents = Vector256.ConvertToInt32(n);
        var bias = Vector256.Create(127);
        var half = exponents >> 1;
        var scaleHigh = ((half + bias) << 23).AsSingle();
        var scaleLow = ((exponents - half + bias) << 23).AsSingle();
        var result = p * scaleHigh * scaleLow;
        result = Vector256.ConditionalSelect(Vector256.LessThan(x, Vector256.Create(Lowest)), Vector256<float>.Zero, result);
        result = Vector256.ConditionalSelect(Vector256.GreaterThan(x, Vector256.Create(Highest)), Vector256.Create(float.PositiveInfinity), result);
        return Vector256.ConditionalSelect(Vector256.Equals(x, x), result, x);
    }

    /// <summary>e^x of four lanes.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> Exp(Vector128<float> x)
    {
        var clamped = Vector128.MinNative(Vector128.MaxNative(x, Vector128.Create(Lowest)), Vector128.Create(Highest));
        var n = Vector128.Round(clamped * Log2E);
        var r = Vector128.FusedMultiplyAdd(n, Vector128.Create(-Ln2Low), Vector128.FusedMultiplyAdd(n, Vector128.Create(-Ln2High), clamped));
        var p = Vector128.FusedMultiplyAdd(Vector128.Create(P0), r, Vector128.Create(P1));
        p = Vector128.FusedMultiplyAdd(p, r, Vector128.Create(P2));
        p = Vector128.FusedMultiplyAdd(p, r, Vector128.Create(P3));
        p = Vector128.FusedMultiplyAdd(p, r, Vector128.Create(P4));
        p = Vector128.FusedMultiplyAdd(p, r, Vector128.Create(P5));
        p = Vector128.FusedMultiplyAdd(p, r * r, r + Vector128<float>.One);
        var exponents = Vector128.ConvertToInt32(n);
        var bias = Vector128.Create(127);
        var half = exponents >> 1;
        var scaleHigh = ((half + bias) << 23).AsSingle();
        var scaleLow = ((exponents - half + bias) << 23).AsSingle();
        var result = p * scaleHigh * scaleLow;
        result = Vector128.ConditionalSelect(Vector128.LessThan(x, Vector128.Create(Lowest)), Vector128<float>.Zero, result);
        result = Vector128.ConditionalSelect(Vector128.GreaterThan(x, Vector128.Create(Highest)), Vector128.Create(float.PositiveInfinity), result);
        return Vector128.ConditionalSelect(Vector128.Equals(x, x), result, x);
    }

    /// <summary>e^x of sixteen lanes.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> Exp(Vector512<float> x)
    {
        var clamped = Vector512.MinNative(Vector512.MaxNative(x, Vector512.Create(Lowest)), Vector512.Create(Highest));
        var n = Vector512.Round(clamped * Log2E);
        var r = Vector512.FusedMultiplyAdd(n, Vector512.Create(-Ln2Low), Vector512.FusedMultiplyAdd(n, Vector512.Create(-Ln2High), clamped));
        var p = Vector512.FusedMultiplyAdd(Vector512.Create(P0), r, Vector512.Create(P1));
        p = Vector512.FusedMultiplyAdd(p, r, Vector512.Create(P2));
        p = Vector512.FusedMultiplyAdd(p, r, Vector512.Create(P3));
        p = Vector512.FusedMultiplyAdd(p, r, Vector512.Create(P4));
        p = Vector512.FusedMultiplyAdd(p, r, Vector512.Create(P5));
        p = Vector512.FusedMultiplyAdd(p, r * r, r + Vector512<float>.One);
        var exponents = Vector512.ConvertToInt32(n);
        var bias = Vector512.Create(127);
        var half = exponents >> 1;
        var scaleHigh = ((half + bias) << 23).AsSingle();
        var scaleLow = ((exponents - half + bias) << 23).AsSingle();
        var result = p * scaleHigh * scaleLow;
        result = Vector512.ConditionalSelect(Vector512.LessThan(x, Vector512.Create(Lowest)), Vector512<float>.Zero, result);
        result = Vector512.ConditionalSelect(Vector512.GreaterThan(x, Vector512.Create(Highest)), Vector512.Create(float.PositiveInfinity), result);
        return Vector512.ConditionalSelect(Vector512.Equals(x, x), result, x);
    }
}
