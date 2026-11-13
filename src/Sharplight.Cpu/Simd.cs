using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Sharplight.Cpu;

/// <summary>Low level SIMD helpers (AVX-512 / AVX2 / scalar fallbacks).</summary>
internal static unsafe class Simd
{
    // ---- float dot products -------------------------------------------------------------------
    public static float Dot(float* a, float* b, long n)
    {
        long i = 0;
        float sum = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            var s0 = Vector512<float>.Zero;
            var s1 = s0;
            var s2 = s0;
            var s3 = s0;
            for (; i + 64 <= n; i += 64)
            {
                s0 = Vector512.FusedMultiplyAdd(Vector512.Load(a + i), Vector512.Load(b + i), s0);
                s1 = Vector512.FusedMultiplyAdd(Vector512.Load(a + i + 16), Vector512.Load(b + i + 16), s1);
                s2 = Vector512.FusedMultiplyAdd(Vector512.Load(a + i + 32), Vector512.Load(b + i + 32), s2);
                s3 = Vector512.FusedMultiplyAdd(Vector512.Load(a + i + 48), Vector512.Load(b + i + 48), s3);
            }
            for (; i + 16 <= n; i += 16)
            {
                s0 = Vector512.FusedMultiplyAdd(Vector512.Load(a + i), Vector512.Load(b + i), s0);
            }

            sum = Vector512.Sum((s0 + s1) + (s2 + s3));
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            var s0 = Vector256<float>.Zero;
            var s1 = s0;
            var s2 = s0;
            var s3 = s0;
            for (; i + 32 <= n; i += 32)
            {
                s0 = Vector256.FusedMultiplyAdd(Vector256.Load(a + i), Vector256.Load(b + i), s0);
                s1 = Vector256.FusedMultiplyAdd(Vector256.Load(a + i + 8), Vector256.Load(b + i + 8), s1);
                s2 = Vector256.FusedMultiplyAdd(Vector256.Load(a + i + 16), Vector256.Load(b + i + 16), s2);
                s3 = Vector256.FusedMultiplyAdd(Vector256.Load(a + i + 24), Vector256.Load(b + i + 24), s3);
            }
            for (; i + 8 <= n; i += 8)
            {
                s0 = Vector256.FusedMultiplyAdd(Vector256.Load(a + i), Vector256.Load(b + i), s0);
            }

            sum = Vector256.Sum((s0 + s1) + (s2 + s3));
        }
        for (; i < n; i++)
        {
            sum += a[i] * b[i];
        }

        return sum;
    }

    /// <summary>Four dot products of <paramref name="x"/> with four weight rows at once.</summary>
    public static void Dot4(float* x, float* w0, float* w1, float* w2, float* w3, long n, float* res)
    {
        long i = 0;
        float r0 = 0;
        float r1 = 0;
        float r2 = 0;
        float r3 = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            var a0 = Vector512<float>.Zero;
            var a1 = a0;
            var a2 = a0;
            var a3 = a0;
            var b0 = a0;
            var b1 = a0;
            var b2 = a0;
            var b3 = a0;
            for (; i + 32 <= n; i += 32)
            {
                var xv = Vector512.Load(x + i);
                var xw = Vector512.Load(x + i + 16);
                a0 = Vector512.FusedMultiplyAdd(Vector512.Load(w0 + i), xv, a0);
                a1 = Vector512.FusedMultiplyAdd(Vector512.Load(w1 + i), xv, a1);
                a2 = Vector512.FusedMultiplyAdd(Vector512.Load(w2 + i), xv, a2);
                a3 = Vector512.FusedMultiplyAdd(Vector512.Load(w3 + i), xv, a3);
                b0 = Vector512.FusedMultiplyAdd(Vector512.Load(w0 + i + 16), xw, b0);
                b1 = Vector512.FusedMultiplyAdd(Vector512.Load(w1 + i + 16), xw, b1);
                b2 = Vector512.FusedMultiplyAdd(Vector512.Load(w2 + i + 16), xw, b2);
                b3 = Vector512.FusedMultiplyAdd(Vector512.Load(w3 + i + 16), xw, b3);
            }
            for (; i + 16 <= n; i += 16)
            {
                var xv = Vector512.Load(x + i);
                a0 = Vector512.FusedMultiplyAdd(Vector512.Load(w0 + i), xv, a0);
                a1 = Vector512.FusedMultiplyAdd(Vector512.Load(w1 + i), xv, a1);
                a2 = Vector512.FusedMultiplyAdd(Vector512.Load(w2 + i), xv, a2);
                a3 = Vector512.FusedMultiplyAdd(Vector512.Load(w3 + i), xv, a3);
            }
            r0 = Vector512.Sum(a0 + b0);
            r1 = Vector512.Sum(a1 + b1);
            r2 = Vector512.Sum(a2 + b2);
            r3 = Vector512.Sum(a3 + b3);
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            var a0 = Vector256<float>.Zero;
            var a1 = a0;
            var a2 = a0;
            var a3 = a0;
            for (; i + 8 <= n; i += 8)
            {
                var xv = Vector256.Load(x + i);
                a0 = Vector256.FusedMultiplyAdd(Vector256.Load(w0 + i), xv, a0);
                a1 = Vector256.FusedMultiplyAdd(Vector256.Load(w1 + i), xv, a1);
                a2 = Vector256.FusedMultiplyAdd(Vector256.Load(w2 + i), xv, a2);
                a3 = Vector256.FusedMultiplyAdd(Vector256.Load(w3 + i), xv, a3);
            }
            r0 = Vector256.Sum(a0);
            r1 = Vector256.Sum(a1);
            r2 = Vector256.Sum(a2);
            r3 = Vector256.Sum(a3);
        }
        for (; i < n; i++)
        {
            float xv = x[i];
            r0 += w0[i] * xv;
            r1 += w1[i] * xv;
            r2 += w2[i] * xv;
            r3 += w3[i] * xv;
        }
        res[0] = r0;
        res[1] = r1;
        res[2] = r2;
        res[3] = r3;
    }

    /// <summary>Two tokens x four weight rows: reuses every weight load for both tokens. res = [t0r0..t0r3, t1r0..t1r3].</summary>
    public static void Dot4x2(float* x0, float* x1, float* w0, float* w1, float* w2, float* w3, long n, float* res)
    {
        long i = 0;
        var r = new Span<float>(res, 8);
        r.Clear();
        if (Vector512.IsHardwareAccelerated)
        {
            var a0 = Vector512<float>.Zero;
            var a1 = a0;
            var a2 = a0;
            var a3 = a0;
            var c0 = a0;
            var c1 = a0;
            var c2 = a0;
            var c3 = a0;
            for (; i + 16 <= n; i += 16)
            {
                var xa = Vector512.Load(x0 + i);
                var xb = Vector512.Load(x1 + i);
                var v0 = Vector512.Load(w0 + i);
                var v1 = Vector512.Load(w1 + i);
                var v2 = Vector512.Load(w2 + i);
                var v3 = Vector512.Load(w3 + i);
                a0 = Vector512.FusedMultiplyAdd(v0, xa, a0);
                c0 = Vector512.FusedMultiplyAdd(v0, xb, c0);
                a1 = Vector512.FusedMultiplyAdd(v1, xa, a1);
                c1 = Vector512.FusedMultiplyAdd(v1, xb, c1);
                a2 = Vector512.FusedMultiplyAdd(v2, xa, a2);
                c2 = Vector512.FusedMultiplyAdd(v2, xb, c2);
                a3 = Vector512.FusedMultiplyAdd(v3, xa, a3);
                c3 = Vector512.FusedMultiplyAdd(v3, xb, c3);
            }
            res[0] = Vector512.Sum(a0);
            res[1] = Vector512.Sum(a1);
            res[2] = Vector512.Sum(a2);
            res[3] = Vector512.Sum(a3);
            res[4] = Vector512.Sum(c0);
            res[5] = Vector512.Sum(c1);
            res[6] = Vector512.Sum(c2);
            res[7] = Vector512.Sum(c3);
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            var a0 = Vector256<float>.Zero;
            var a1 = a0;
            var a2 = a0;
            var a3 = a0;
            var c0 = a0;
            var c1 = a0;
            var c2 = a0;
            var c3 = a0;
            for (; i + 8 <= n; i += 8)
            {
                var xa = Vector256.Load(x0 + i);
                var xb = Vector256.Load(x1 + i);
                var v0 = Vector256.Load(w0 + i);
                var v1 = Vector256.Load(w1 + i);
                var v2 = Vector256.Load(w2 + i);
                var v3 = Vector256.Load(w3 + i);
                a0 = Vector256.FusedMultiplyAdd(v0, xa, a0);
                c0 = Vector256.FusedMultiplyAdd(v0, xb, c0);
                a1 = Vector256.FusedMultiplyAdd(v1, xa, a1);
                c1 = Vector256.FusedMultiplyAdd(v1, xb, c1);
                a2 = Vector256.FusedMultiplyAdd(v2, xa, a2);
                c2 = Vector256.FusedMultiplyAdd(v2, xb, c2);
                a3 = Vector256.FusedMultiplyAdd(v3, xa, a3);
                c3 = Vector256.FusedMultiplyAdd(v3, xb, c3);
            }
            res[0] = Vector256.Sum(a0);
            res[1] = Vector256.Sum(a1);
            res[2] = Vector256.Sum(a2);
            res[3] = Vector256.Sum(a3);
            res[4] = Vector256.Sum(c0);
            res[5] = Vector256.Sum(c1);
            res[6] = Vector256.Sum(c2);
            res[7] = Vector256.Sum(c3);
        }
        for (; i < n; i++)
        {
            float xa = x0[i];
            float xb = x1[i];
            res[0] += w0[i] * xa;
            res[1] += w1[i] * xa;
            res[2] += w2[i] * xa;
            res[3] += w3[i] * xa;
            res[4] += w0[i] * xb;
            res[5] += w1[i] * xb;
            res[6] += w2[i] * xb;
            res[7] += w3[i] * xb;
        }
    }

    public static float SumSq(float* a, long n)
    {
        long i = 0;
        float sum = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            var s0 = Vector512<float>.Zero;
            var s1 = s0;
            for (; i + 32 <= n; i += 32)
            {
                var v0 = Vector512.Load(a + i);
                var v1 = Vector512.Load(a + i + 16);
                s0 = Vector512.FusedMultiplyAdd(v0, v0, s0);
                s1 = Vector512.FusedMultiplyAdd(v1, v1, s1);
            }
            for (; i + 16 <= n; i += 16)
            {
                var v = Vector512.Load(a + i);
                s0 = Vector512.FusedMultiplyAdd(v, v, s0);
            }
            sum = Vector512.Sum(s0 + s1);
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            var s0 = Vector256<float>.Zero;
            for (; i + 8 <= n; i += 8)
            {
                var v = Vector256.Load(a + i);
                s0 = Vector256.FusedMultiplyAdd(v, v, s0);
            }
            sum = Vector256.Sum(s0);
        }
        for (; i < n; i++)
        {
            sum += a[i] * a[i];
        }

        return sum;
    }

    public static float Sum(float* a, long n)
    {
        long i = 0;
        float sum = 0;
        if (Vector256.IsHardwareAccelerated)
        {
            var s = Vector256<float>.Zero;
            for (; i + 8 <= n; i += 8)
            {
                s += Vector256.Load(a + i);
            }

            sum = Vector256.Sum(s);
        }
        for (; i < n; i++)
        {
            sum += a[i];
        }

        return sum;
    }

    public static float Max(float* a, long n)
    {
        long i = 0;
        float m = float.NegativeInfinity;
        if (Vector256.IsHardwareAccelerated && n >= 8)
        {
            var s = Vector256.Create(float.NegativeInfinity);
            for (; i + 8 <= n; i += 8)
            {
                s = Vector256.Max(s, Vector256.Load(a + i));
            }

            var h = Vector128.Max(s.GetLower(), s.GetUpper());
            m = MathF.Max(MathF.Max(h.GetElement(0), h.GetElement(1)), MathF.Max(h.GetElement(2), h.GetElement(3)));
        }
        for (; i < n; i++)
        {
            if (a[i] > m)
            {
                m = a[i];
            }
        }

        return m;
    }

    /// <summary>dst[i] = src[i] * s</summary>
    public static void Scale(float* src, float* dst, float s, long n)
    {
        long i = 0;
        if (Vector256.IsHardwareAccelerated)
        {
            var sv = Vector256.Create(s);
            for (; i + 8 <= n; i += 8)
            {
                (Vector256.Load(src + i) * sv).Store(dst + i);
            }
        }
        for (; i < n; i++)
        {
            dst[i] = src[i] * s;
        }
    }

    /// <summary>dst[i] = src[i] * s * w[i]</summary>
    public static void ScaleMul(float* src, float* w, float* dst, float s, long n)
    {
        long i = 0;
        if (Vector256.IsHardwareAccelerated)
        {
            var sv = Vector256.Create(s);
            for (; i + 8 <= n; i += 8)
            {
                (Vector256.Load(src + i) * sv * Vector256.Load(w + i)).Store(dst + i);
            }
        }
        for (; i < n; i++)
        {
            dst[i] = src[i] * s * w[i];
        }
    }

    /// <summary>acc[i] += p * v[i]</summary>
    public static void Axpy(float* acc, float* v, float p, long n)
    {
        long i = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            var pv = Vector512.Create(p);
            for (; i + 16 <= n; i += 16)
            {
                Vector512.FusedMultiplyAdd(Vector512.Load(v + i), pv, Vector512.Load(acc + i)).Store(acc + i);
            }
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            var pv = Vector256.Create(p);
            for (; i + 8 <= n; i += 8)
            {
                Vector256.FusedMultiplyAdd(Vector256.Load(v + i), pv, Vector256.Load(acc + i)).Store(acc + i);
            }
        }
        for (; i < n; i++)
        {
            acc[i] += p * v[i];
        }
    }

    // ---- half / bfloat conversion -------------------------------------------------------------
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float HalfToFloat(ushort h) => (float)Unsafe.As<ushort, Half>(ref h);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ushort FloatToHalf(float f)
    {
        Half h = (Half)f;
        return Unsafe.As<Half, ushort>(ref h);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float BFloatToFloat(ushort h) => BitConverter.UInt32BitsToSingle((uint)h << 16);

    public static ushort FloatToBFloat(float f)
    {
        uint b = BitConverter.SingleToUInt32Bits(f);
        if ((b & 0x7FFFFFFF) > 0x7F800000)
        {
            return (ushort)((b >> 16) | 0x40);
        }

        b += 0x7FFF + ((b >> 16) & 1);
        return (ushort)(b >> 16);
    }

    /// <summary>Converts 16 half values (as widened uint lanes) to float.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<float> HalfBitsToSingle(Vector512<uint> h)
    {
        var sign = Vector512.ShiftLeft(h & Vector512.Create(0x8000u), 16);
        var shifted = Vector512.ShiftLeft(h & Vector512.Create(0x7FFFu), 13);
        var f = shifted.AsSingle() * Vector512.Create(BitConverter.UInt32BitsToSingle(0x77800000u));
        var infnan = Vector512.GreaterThanOrEqual(f, Vector512.Create(65536f)).AsUInt32() & Vector512.Create(0x7F800000u);
        return (f.AsUInt32() | infnan | sign).AsSingle();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> HalfBitsToSingle(Vector256<uint> h)
    {
        var sign = Vector256.ShiftLeft(h & Vector256.Create(0x8000u), 16);
        var shifted = Vector256.ShiftLeft(h & Vector256.Create(0x7FFFu), 13);
        var f = shifted.AsSingle() * Vector256.Create(BitConverter.UInt32BitsToSingle(0x77800000u));
        var infnan = Vector256.GreaterThanOrEqual(f, Vector256.Create(65536f)).AsUInt32() & Vector256.Create(0x7F800000u);
        return (f.AsUInt32() | infnan | sign).AsSingle();
    }

    public static void HalfToFloat(ushort* src, float* dst, long n)
    {
        long i = 0;
        if (Avx512F.IsSupported)
        {
            for (; i + 16 <= n; i += 16)
            {
                HalfBitsToSingle(Avx512F.ConvertToVector512UInt32(Vector256.Load(src + i))).Store(dst + i);
            }
        }
        else if (Avx2.IsSupported)
        {
            for (; i + 8 <= n; i += 8)
            {
                HalfBitsToSingle(Avx2.ConvertToVector256Int32(Vector128.Load(src + i)).AsUInt32()).Store(dst + i);
            }
        }

        for (; i < n; i++)
        {
            dst[i] = HalfToFloat(src[i]);
        }
    }

    public static void BFloatToFloat(ushort* src, float* dst, long n)
    {
        long i = 0;
        if (Avx512F.IsSupported)
        {
            for (; i + 16 <= n; i += 16)
            {
                Vector512.ShiftLeft(Avx512F.ConvertToVector512UInt32(Vector256.Load(src + i)), 16).AsSingle().Store(dst + i);
            }
        }
        else if (Avx2.IsSupported)
        {
            for (; i + 8 <= n; i += 8)
            {
                Vector256.ShiftLeft(Avx2.ConvertToVector256Int32(Vector128.Load(src + i)).AsUInt32(), 16).AsSingle().Store(dst + i);
            }
        }

        for (; i < n; i++)
        {
            dst[i] = BFloatToFloat(src[i]);
        }
    }

    /// <summary>Dot product of f32 vector with half vector.</summary>
    public static float DotHalf(float* a, ushort* h, long n)
    {
        long i = 0;
        float sum = 0;
        if (Avx512F.IsSupported)
        {
            var s0 = Vector512<float>.Zero;
            var s1 = s0;
            for (; i + 32 <= n; i += 32)
            {
                s0 = Vector512.FusedMultiplyAdd(Vector512.Load(a + i), HalfBitsToSingle(Avx512F.ConvertToVector512UInt32(Vector256.Load(h + i))), s0);
                s1 = Vector512.FusedMultiplyAdd(Vector512.Load(a + i + 16), HalfBitsToSingle(Avx512F.ConvertToVector512UInt32(Vector256.Load(h + i + 16))), s1);
            }
            for (; i + 16 <= n; i += 16)
            {
                s0 = Vector512.FusedMultiplyAdd(Vector512.Load(a + i), HalfBitsToSingle(Avx512F.ConvertToVector512UInt32(Vector256.Load(h + i))), s0);
            }

            sum = Vector512.Sum(s0 + s1);
        }
        else if (Avx2.IsSupported)
        {
            var s0 = Vector256<float>.Zero;
            for (; i + 8 <= n; i += 8)
            {
                s0 = Vector256.FusedMultiplyAdd(Vector256.Load(a + i), HalfBitsToSingle(Avx2.ConvertToVector256Int32(Vector128.Load(h + i)).AsUInt32()), s0);
            }

            sum = Vector256.Sum(s0);
        }
        for (; i < n; i++)
        {
            sum += a[i] * HalfToFloat(h[i]);
        }

        return sum;
    }

    /// <summary>acc[i] += p * half[i]</summary>
    public static void AxpyHalf(float* acc, ushort* h, float p, long n)
    {
        long i = 0;
        if (Avx512F.IsSupported)
        {
            var pv = Vector512.Create(p);
            for (; i + 16 <= n; i += 16)
            {
                Vector512.FusedMultiplyAdd(HalfBitsToSingle(Avx512F.ConvertToVector512UInt32(Vector256.Load(h + i))), pv, Vector512.Load(acc + i)).Store(acc + i);
            }
        }
        else if (Avx2.IsSupported)
        {
            var pv = Vector256.Create(p);
            for (; i + 8 <= n; i += 8)
            {
                Vector256.FusedMultiplyAdd(HalfBitsToSingle(Avx2.ConvertToVector256Int32(Vector128.Load(h + i)).AsUInt32()), pv, Vector256.Load(acc + i)).Store(acc + i);
            }
        }
        for (; i < n; i++)
        {
            acc[i] += p * HalfToFloat(h[i]);
        }
    }

    // ---- small-integer widening ----------------------------------------------------------------
    /// <summary>dst[0..16) = u8[i] * scale + offset</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void StoreU8x16(Vector128<byte> v, float scale, float offset, float* dst)
    {
        if (Avx512F.IsSupported)
        {
            var f = Vector512.ConvertToSingle(Avx512F.ConvertToVector512Int32(v));
            Vector512.FusedMultiplyAdd(f, Vector512.Create(scale), Vector512.Create(offset)).Store(dst);
        }
        else if (Avx2.IsSupported)
        {
            var sv = Vector256.Create(scale);
            var ov = Vector256.Create(offset);
            Vector256.FusedMultiplyAdd(Vector256.ConvertToSingle(Avx2.ConvertToVector256Int32(v)), sv, ov).Store(dst);
            Vector256.FusedMultiplyAdd(Vector256.ConvertToSingle(Avx2.ConvertToVector256Int32(Sse2.ShiftRightLogical128BitLane(v, 8))), sv, ov).Store(dst + 8);
        }
        else
        {
            for (int i = 0; i < 16; i++)
            {
                dst[i] = v.GetElement(i) * scale + offset;
            }
        }
    }

    /// <summary>dst[0..16) = i8[i] * scale + offset</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void StoreI8x16(Vector128<sbyte> v, float scale, float offset, float* dst)
    {
        if (Avx512F.IsSupported)
        {
            var f = Vector512.ConvertToSingle(Avx512F.ConvertToVector512Int32(v));
            Vector512.FusedMultiplyAdd(f, Vector512.Create(scale), Vector512.Create(offset)).Store(dst);
        }
        else if (Avx2.IsSupported)
        {
            var sv = Vector256.Create(scale);
            var ov = Vector256.Create(offset);
            Vector256.FusedMultiplyAdd(Vector256.ConvertToSingle(Avx2.ConvertToVector256Int32(v)), sv, ov).Store(dst);
            Vector256.FusedMultiplyAdd(Vector256.ConvertToSingle(Avx2.ConvertToVector256Int32(Sse2.ShiftRightLogical128BitLane(v, 8))), sv, ov).Store(dst + 8);
        }
        else
        {
            for (int i = 0; i < 16; i++)
            {
                dst[i] = v.GetElement(i) * scale + offset;
            }
        }
    }
}
