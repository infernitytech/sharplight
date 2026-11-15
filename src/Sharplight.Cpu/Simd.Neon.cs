using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;

namespace Sharplight.Cpu;

// Arm64 NEON tiers of the SIMD helpers (CpuIsa.UseNeon): 128-bit vectors with enough independent accumulators to cover
// the four-cycle FMA latency of the wide Arm cores. Each helper processes whole vectors from *i on and leaves *i at the
// first element of the scalar tail, which the caller finishes as on every other tier.
internal static unsafe partial class Simd
{
    private static float DotNeon(float* a, float* b, long n, ref long i)
    {
        var s0 = Vector128<float>.Zero;
        var s1 = s0;
        var s2 = s0;
        var s3 = s0;
        for (; i + 16 <= n; i += 16)
        {
            s0 = AdvSimd.FusedMultiplyAdd(s0, Vector128.Load(a + i), Vector128.Load(b + i));
            s1 = AdvSimd.FusedMultiplyAdd(s1, Vector128.Load(a + i + 4), Vector128.Load(b + i + 4));
            s2 = AdvSimd.FusedMultiplyAdd(s2, Vector128.Load(a + i + 8), Vector128.Load(b + i + 8));
            s3 = AdvSimd.FusedMultiplyAdd(s3, Vector128.Load(a + i + 12), Vector128.Load(b + i + 12));
        }

        for (; i + 4 <= n; i += 4)
        {
            s0 = AdvSimd.FusedMultiplyAdd(s0, Vector128.Load(a + i), Vector128.Load(b + i));
        }

        return Vector128.Sum((s0 + s1) + (s2 + s3));
    }

    private static void Dot4Neon(float* x, float* w0, float* w1, float* w2, float* w3, long n, ref long i, float* res)
    {
        var a0 = Vector128<float>.Zero;
        var a1 = a0;
        var a2 = a0;
        var a3 = a0;
        var b0 = a0;
        var b1 = a0;
        var b2 = a0;
        var b3 = a0;
        for (; i + 8 <= n; i += 8)
        {
            var xv = Vector128.Load(x + i);
            var xw = Vector128.Load(x + i + 4);
            a0 = AdvSimd.FusedMultiplyAdd(a0, Vector128.Load(w0 + i), xv);
            a1 = AdvSimd.FusedMultiplyAdd(a1, Vector128.Load(w1 + i), xv);
            a2 = AdvSimd.FusedMultiplyAdd(a2, Vector128.Load(w2 + i), xv);
            a3 = AdvSimd.FusedMultiplyAdd(a3, Vector128.Load(w3 + i), xv);
            b0 = AdvSimd.FusedMultiplyAdd(b0, Vector128.Load(w0 + i + 4), xw);
            b1 = AdvSimd.FusedMultiplyAdd(b1, Vector128.Load(w1 + i + 4), xw);
            b2 = AdvSimd.FusedMultiplyAdd(b2, Vector128.Load(w2 + i + 4), xw);
            b3 = AdvSimd.FusedMultiplyAdd(b3, Vector128.Load(w3 + i + 4), xw);
        }

        for (; i + 4 <= n; i += 4)
        {
            var xv = Vector128.Load(x + i);
            a0 = AdvSimd.FusedMultiplyAdd(a0, Vector128.Load(w0 + i), xv);
            a1 = AdvSimd.FusedMultiplyAdd(a1, Vector128.Load(w1 + i), xv);
            a2 = AdvSimd.FusedMultiplyAdd(a2, Vector128.Load(w2 + i), xv);
            a3 = AdvSimd.FusedMultiplyAdd(a3, Vector128.Load(w3 + i), xv);
        }

        res[0] = Vector128.Sum(a0 + b0);
        res[1] = Vector128.Sum(a1 + b1);
        res[2] = Vector128.Sum(a2 + b2);
        res[3] = Vector128.Sum(a3 + b3);
    }

    private static void Dot4x2Neon(float* x0, float* x1, float* w0, float* w1, float* w2, float* w3, long n, ref long i, float* res)
    {
        var a0 = Vector128<float>.Zero;
        var a1 = a0;
        var a2 = a0;
        var a3 = a0;
        var c0 = a0;
        var c1 = a0;
        var c2 = a0;
        var c3 = a0;
        for (; i + 4 <= n; i += 4)
        {
            var xa = Vector128.Load(x0 + i);
            var xb = Vector128.Load(x1 + i);
            var v0 = Vector128.Load(w0 + i);
            var v1 = Vector128.Load(w1 + i);
            var v2 = Vector128.Load(w2 + i);
            var v3 = Vector128.Load(w3 + i);
            a0 = AdvSimd.FusedMultiplyAdd(a0, v0, xa);
            c0 = AdvSimd.FusedMultiplyAdd(c0, v0, xb);
            a1 = AdvSimd.FusedMultiplyAdd(a1, v1, xa);
            c1 = AdvSimd.FusedMultiplyAdd(c1, v1, xb);
            a2 = AdvSimd.FusedMultiplyAdd(a2, v2, xa);
            c2 = AdvSimd.FusedMultiplyAdd(c2, v2, xb);
            a3 = AdvSimd.FusedMultiplyAdd(a3, v3, xa);
            c3 = AdvSimd.FusedMultiplyAdd(c3, v3, xb);
        }

        res[0] = Vector128.Sum(a0);
        res[1] = Vector128.Sum(a1);
        res[2] = Vector128.Sum(a2);
        res[3] = Vector128.Sum(a3);
        res[4] = Vector128.Sum(c0);
        res[5] = Vector128.Sum(c1);
        res[6] = Vector128.Sum(c2);
        res[7] = Vector128.Sum(c3);
    }

    private static float SumSqNeon(float* a, long n, ref long i)
    {
        var s0 = Vector128<float>.Zero;
        var s1 = s0;
        for (; i + 8 <= n; i += 8)
        {
            var v0 = Vector128.Load(a + i);
            var v1 = Vector128.Load(a + i + 4);
            s0 = AdvSimd.FusedMultiplyAdd(s0, v0, v0);
            s1 = AdvSimd.FusedMultiplyAdd(s1, v1, v1);
        }

        for (; i + 4 <= n; i += 4)
        {
            var v = Vector128.Load(a + i);
            s0 = AdvSimd.FusedMultiplyAdd(s0, v, v);
        }

        return Vector128.Sum(s0 + s1);
    }

    private static float SumNeon(float* a, long n, ref long i)
    {
        var s0 = Vector128<float>.Zero;
        var s1 = s0;
        for (; i + 8 <= n; i += 8)
        {
            s0 += Vector128.Load(a + i);
            s1 += Vector128.Load(a + i + 4);
        }

        for (; i + 4 <= n; i += 4)
        {
            s0 += Vector128.Load(a + i);
        }

        return Vector128.Sum(s0 + s1);
    }

    private static float MaxNeon(float* a, long n, ref long i)
    {
        var s = Vector128.Create(float.NegativeInfinity);
        for (; i + 4 <= n; i += 4)
        {
            s = AdvSimd.Max(s, Vector128.Load(a + i));
        }

        return AdvSimd.Arm64.MaxAcross(s).ToScalar();
    }

    private static float ExpShiftedNeon(float* x, long n, float shift, ref long i)
    {
        var sv = Vector128.Create(shift);
        var acc = Vector128<float>.Zero;
        for (; i + 4 <= n; i += 4)
        {
            var e = Vector128.Exp(Vector128.Load(x + i) - sv);
            e.Store(x + i);
            acc += e;
        }

        return Vector128.Sum(acc);
    }

    private static void ScaleNeon(float* src, float* dst, float s, long n, ref long i)
    {
        var sv = Vector128.Create(s);
        for (; i + 4 <= n; i += 4)
        {
            (Vector128.Load(src + i) * sv).Store(dst + i);
        }
    }

    private static void ScaleMulNeon(float* src, float* w, float* dst, float s, long n, ref long i)
    {
        var sv = Vector128.Create(s);
        for (; i + 4 <= n; i += 4)
        {
            (Vector128.Load(src + i) * sv * Vector128.Load(w + i)).Store(dst + i);
        }
    }

    private static void AxpyNeon(float* acc, float* v, float p, long n, ref long i)
    {
        var pv = Vector128.Create(p);
        for (; i + 4 <= n; i += 4)
        {
            AdvSimd.FusedMultiplyAdd(Vector128.Load(acc + i), Vector128.Load(v + i), pv).Store(acc + i);
        }
    }

    /// <summary>Converts 4 half values (widened to uint lanes) to float; the same bit manipulation as the x86 tiers.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> HalfBitsToSingle(Vector128<uint> h)
    {
        var sign = Vector128.ShiftLeft(h & Vector128.Create(0x8000u), 16);
        var shifted = Vector128.ShiftLeft(h & Vector128.Create(0x7FFFu), 13);
        var f = shifted.AsSingle() * Vector128.Create(BitConverter.UInt32BitsToSingle(0x77800000u));
        var infnan = Vector128.GreaterThanOrEqual(f, Vector128.Create(65536f)).AsUInt32() & Vector128.Create(0x7F800000u);
        return (f.AsUInt32() | infnan | sign).AsSingle();
    }

    /// <summary>Eight half values at <paramref name="h"/> as two float vectors.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static (Vector128<float> Lower, Vector128<float> Upper) LoadHalf8(ushort* h)
    {
        var (lo, hi) = Vector128.Widen(Vector128.Load(h));
        return (HalfBitsToSingle(lo), HalfBitsToSingle(hi));
    }

    private static void HalfToFloatNeon(ushort* src, float* dst, long n, ref long i)
    {
        for (; i + 8 <= n; i += 8)
        {
            var (lo, hi) = LoadHalf8(src + i);
            lo.Store(dst + i);
            hi.Store(dst + i + 4);
        }
    }

    private static void BFloatToFloatNeon(ushort* src, float* dst, long n, ref long i)
    {
        for (; i + 8 <= n; i += 8)
        {
            var (lo, hi) = Vector128.Widen(Vector128.Load(src + i));
            Vector128.ShiftLeft(lo, 16).AsSingle().Store(dst + i);
            Vector128.ShiftLeft(hi, 16).AsSingle().Store(dst + i + 4);
        }
    }

    private static float DotHalfNeon(float* a, ushort* h, long n, ref long i)
    {
        var s0 = Vector128<float>.Zero;
        var s1 = s0;
        var s2 = s0;
        var s3 = s0;
        for (; i + 16 <= n; i += 16)
        {
            var (h0, h1) = LoadHalf8(h + i);
            var (h2, h3) = LoadHalf8(h + i + 8);
            s0 = AdvSimd.FusedMultiplyAdd(s0, Vector128.Load(a + i), h0);
            s1 = AdvSimd.FusedMultiplyAdd(s1, Vector128.Load(a + i + 4), h1);
            s2 = AdvSimd.FusedMultiplyAdd(s2, Vector128.Load(a + i + 8), h2);
            s3 = AdvSimd.FusedMultiplyAdd(s3, Vector128.Load(a + i + 12), h3);
        }

        for (; i + 8 <= n; i += 8)
        {
            var (h0, h1) = LoadHalf8(h + i);
            s0 = AdvSimd.FusedMultiplyAdd(s0, Vector128.Load(a + i), h0);
            s1 = AdvSimd.FusedMultiplyAdd(s1, Vector128.Load(a + i + 4), h1);
        }

        return Vector128.Sum((s0 + s1) + (s2 + s3));
    }

    private static void AxpyHalfNeon(float* acc, ushort* h, float p, long n, ref long i)
    {
        var pv = Vector128.Create(p);
        for (; i + 8 <= n; i += 8)
        {
            var (h0, h1) = LoadHalf8(h + i);
            AdvSimd.FusedMultiplyAdd(Vector128.Load(acc + i), h0, pv).Store(acc + i);
            AdvSimd.FusedMultiplyAdd(Vector128.Load(acc + i + 4), h1, pv).Store(acc + i + 4);
        }
    }

    /// <summary>dst[0..16) = u8[i] * scale + offset.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StoreU8x16Neon(Vector128<byte> v, float scale, float offset, float* dst)
    {
        var sv = Vector128.Create(scale);
        var ov = Vector128.Create(offset);
        var (lo, hi) = Vector128.Widen(v);
        var (l0, l1) = Vector128.Widen(lo);
        var (h0, h1) = Vector128.Widen(hi);
        AdvSimd.FusedMultiplyAdd(ov, Vector128.ConvertToSingle(l0), sv).Store(dst);
        AdvSimd.FusedMultiplyAdd(ov, Vector128.ConvertToSingle(l1), sv).Store(dst + 4);
        AdvSimd.FusedMultiplyAdd(ov, Vector128.ConvertToSingle(h0), sv).Store(dst + 8);
        AdvSimd.FusedMultiplyAdd(ov, Vector128.ConvertToSingle(h1), sv).Store(dst + 12);
    }

    /// <summary>dst[0..16) = i8[i] * scale + offset.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StoreI8x16Neon(Vector128<sbyte> v, float scale, float offset, float* dst)
    {
        var sv = Vector128.Create(scale);
        var ov = Vector128.Create(offset);
        var (lo, hi) = Vector128.Widen(v);
        var (l0, l1) = Vector128.Widen(lo);
        var (h0, h1) = Vector128.Widen(hi);
        AdvSimd.FusedMultiplyAdd(ov, Vector128.ConvertToSingle(l0), sv).Store(dst);
        AdvSimd.FusedMultiplyAdd(ov, Vector128.ConvertToSingle(l1), sv).Store(dst + 4);
        AdvSimd.FusedMultiplyAdd(ov, Vector128.ConvertToSingle(h0), sv).Store(dst + 8);
        AdvSimd.FusedMultiplyAdd(ov, Vector128.ConvertToSingle(h1), sv).Store(dst + 12);
    }
}
