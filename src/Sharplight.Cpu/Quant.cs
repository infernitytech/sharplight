using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using Sharplight.Tensors;

namespace Sharplight.Cpu;

/// <summary>Weight storage formats the CPU provider can read.</summary>
public enum WeightFormat { F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K }

/// <summary>Dequantization of ggml block formats (exact ggml layouts).</summary>
public static unsafe class Quant
{
    public static WeightFormat? FormatOf(DataType t)
    {
        if (t == DataType.F32) return WeightFormat.F32;
        if (t == DataType.F16) return WeightFormat.F16;
        if (t == DataType.BF16) return WeightFormat.BF16;
        if (t == DataType.Q4_0) return WeightFormat.Q4_0;
        if (t == DataType.Q4_1) return WeightFormat.Q4_1;
        if (t == DataType.Q5_0) return WeightFormat.Q5_0;
        if (t == DataType.Q5_1) return WeightFormat.Q5_1;
        if (t == DataType.Q8_0) return WeightFormat.Q8_0;
        if (t == DataType.Q4_K) return WeightFormat.Q4_K;
        if (t == DataType.Q5_K) return WeightFormat.Q5_K;
        if (t == DataType.Q6_K) return WeightFormat.Q6_K;
        return null;
    }

    public static int BlockSize(WeightFormat f) => f switch
    {
        WeightFormat.Q4_K or WeightFormat.Q5_K or WeightFormat.Q6_K => 256,
        WeightFormat.Q4_0 or WeightFormat.Q4_1 or WeightFormat.Q5_0 or WeightFormat.Q5_1 or WeightFormat.Q8_0 => 32,
        _ => 1,
    };

    public static int BlockBytes(WeightFormat f) => f switch
    {
        WeightFormat.F32 => 4,
        WeightFormat.F16 or WeightFormat.BF16 => 2,
        WeightFormat.Q4_0 => 18,
        WeightFormat.Q4_1 => 20,
        WeightFormat.Q5_0 => 22,
        WeightFormat.Q5_1 => 24,
        WeightFormat.Q8_0 => 34,
        WeightFormat.Q4_K => 144,
        WeightFormat.Q5_K => 176,
        WeightFormat.Q6_K => 210,
        _ => throw new ArgumentOutOfRangeException(nameof(f)),
    };

    /// <summary>
    /// Dequantizes <paramref name="count"/> elements starting at a block boundary. <paramref name="src"/> points at the
    /// first block to convert; <paramref name="count"/> must be a multiple of the block size.
    /// </summary>
    public static void Dequantize(WeightFormat f, byte* src, long count, float* dst)
    {
        long bs = BlockSize(f);
        long nb = count / bs;
        switch (f)
        {
            case WeightFormat.F32: Buffer.MemoryCopy(src, dst, count * 4, count * 4); break;
            case WeightFormat.F16: Simd.HalfToFloat((ushort*)src, dst, count); break;
            case WeightFormat.BF16: Simd.BFloatToFloat((ushort*)src, dst, count); break;
            case WeightFormat.Q4_0: Q4_0(src, nb, dst); break;
            case WeightFormat.Q4_1: Q4_1(src, nb, dst); break;
            case WeightFormat.Q5_0: Q5_0(src, nb, dst); break;
            case WeightFormat.Q5_1: Q5_1(src, nb, dst); break;
            case WeightFormat.Q8_0: Q8_0(src, nb, dst); break;
            case WeightFormat.Q4_K: Q4_K(src, nb, dst); break;
            case WeightFormat.Q5_K: Q5_K(src, nb, dst); break;
            case WeightFormat.Q6_K: Q6_K(src, nb, dst); break;
            default: throw new NotSupportedException(f.ToString());
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float H(byte* p)
    {
        uint h = Unsafe.ReadUnaligned<ushort>(p);
        uint e = h & 0x7C00;
        if (e == 0 || e == 0x7C00) return Simd.HalfToFloat((ushort)h); // zero / subnormal / inf / nan
        return BitConverter.UInt32BitsToSingle(((h & 0x8000) << 16) | (((h & 0x7FFF) << 13) + 0x38000000));
    }

    static void Q8_0(byte* p, long nb, float* dst)
    {
        for (long b = 0; b < nb; b++, p += 34, dst += 32)
        {
            float d = H(p);
            Simd.StoreI8x16(Vector128.Load((sbyte*)(p + 2)), d, 0f, dst);
            Simd.StoreI8x16(Vector128.Load((sbyte*)(p + 18)), d, 0f, dst + 16);
        }
    }

    static void Q4_0(byte* p, long nb, float* dst)
    {
        var mask = Vector128.Create((byte)0x0F);
        for (long b = 0; b < nb; b++, p += 18, dst += 32)
        {
            float d = H(p);
            var q = Vector128.Load(p + 2);
            Simd.StoreU8x16(q & mask, d, -8f * d, dst);
            Simd.StoreU8x16(Vector128.ShiftRightLogical(q, 4), d, -8f * d, dst + 16);
        }
    }

    static void Q4_1(byte* p, long nb, float* dst)
    {
        var mask = Vector128.Create((byte)0x0F);
        for (long b = 0; b < nb; b++, p += 20, dst += 32)
        {
            float d = H(p), m = H(p + 2);
            var q = Vector128.Load(p + 4);
            Simd.StoreU8x16(q & mask, d, m, dst);
            Simd.StoreU8x16(Vector128.ShiftRightLogical(q, 4), d, m, dst + 16);
        }
    }

    static void Q5_0(byte* p, long nb, float* dst)
    {
        var mask = Vector128.Create((byte)0x0F);
        byte* hb = stackalloc byte[32];
        for (long b = 0; b < nb; b++, p += 22, dst += 32)
        {
            float d = H(p);
            uint qh = Unsafe.ReadUnaligned<uint>(p + 2);
            for (int j = 0; j < 16; j++) { hb[j] = (byte)(((qh >> j) & 1) << 4); hb[j + 16] = (byte)(((qh >> (j + 16)) & 1) << 4); }
            var q = Vector128.Load(p + 6);
            Simd.StoreU8x16((q & mask) | Vector128.Load(hb), d, -16f * d, dst);
            Simd.StoreU8x16(Vector128.ShiftRightLogical(q, 4) | Vector128.Load(hb + 16), d, -16f * d, dst + 16);
        }
    }

    static void Q5_1(byte* p, long nb, float* dst)
    {
        var mask = Vector128.Create((byte)0x0F);
        byte* hb = stackalloc byte[32];
        for (long b = 0; b < nb; b++, p += 24, dst += 32)
        {
            float d = H(p), m = H(p + 2);
            uint qh = Unsafe.ReadUnaligned<uint>(p + 4);
            for (int j = 0; j < 16; j++) { hb[j] = (byte)(((qh >> j) & 1) << 4); hb[j + 16] = (byte)(((qh >> (j + 16)) & 1) << 4); }
            var q = Vector128.Load(p + 8);
            Simd.StoreU8x16((q & mask) | Vector128.Load(hb), d, m, dst);
            Simd.StoreU8x16(Vector128.ShiftRightLogical(q, 4) | Vector128.Load(hb + 16), d, m, dst + 16);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static void ScaleMin(int j, byte* q, out int sc, out int m)
    {
        if (j < 4) { sc = q[j] & 63; m = q[j + 4] & 63; }
        else
        {
            sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
            m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        }
    }

    static void Q4_K(byte* p, long nb, float* dst)
    {
        var mask = Vector128.Create((byte)0x0F);
        for (long b = 0; b < nb; b++, p += 144, dst += 256)
        {
            float d = H(p), dmin = H(p + 2);
            byte* scales = p + 4;
            byte* qs = p + 16;
            for (int j = 0; j < 4; j++, qs += 32, dst += 64)
            {
                ScaleMin(2 * j, scales, out int sc1, out int m1);
                ScaleMin(2 * j + 1, scales, out int sc2, out int m2);
                float d1 = d * sc1, mm1 = dmin * m1, d2 = d * sc2, mm2 = dmin * m2;
                var a0 = Vector128.Load(qs); var a1 = Vector128.Load(qs + 16);
                Simd.StoreU8x16(a0 & mask, d1, -mm1, dst);
                Simd.StoreU8x16(a1 & mask, d1, -mm1, dst + 16);
                Simd.StoreU8x16(Vector128.ShiftRightLogical(a0, 4), d2, -mm2, dst + 32);
                Simd.StoreU8x16(Vector128.ShiftRightLogical(a1, 4), d2, -mm2, dst + 48);
            }
            dst -= 256;
        }
    }

    static void Q5_K(byte* p, long nb, float* dst)
    {
        var mask = Vector128.Create((byte)0x0F);
        var one = Vector128.Create((byte)1);
        for (long b = 0; b < nb; b++, p += 176, dst += 256)
        {
            float d = H(p), dmin = H(p + 2);
            byte* scales = p + 4;
            byte* qh = p + 16;
            byte* ql = p + 48;
            var h0 = Vector128.Load(qh); var h1 = Vector128.Load(qh + 16);
            for (int j = 0; j < 4; j++, ql += 32, dst += 64)
            {
                ScaleMin(2 * j, scales, out int sc1, out int m1);
                ScaleMin(2 * j + 1, scales, out int sc2, out int m2);
                float d1 = d * sc1, mm1 = dmin * m1, d2 = d * sc2, mm2 = dmin * m2;
                var a0 = Vector128.Load(ql); var a1 = Vector128.Load(ql + 16);
                var hl0 = Vector128.ShiftLeft(Vector128.ShiftRightLogical(h0, 2 * j) & one, 4);
                var hl1 = Vector128.ShiftLeft(Vector128.ShiftRightLogical(h1, 2 * j) & one, 4);
                var hh0 = Vector128.ShiftLeft(Vector128.ShiftRightLogical(h0, 2 * j + 1) & one, 4);
                var hh1 = Vector128.ShiftLeft(Vector128.ShiftRightLogical(h1, 2 * j + 1) & one, 4);
                Simd.StoreU8x16((a0 & mask) | hl0, d1, -mm1, dst);
                Simd.StoreU8x16((a1 & mask) | hl1, d1, -mm1, dst + 16);
                Simd.StoreU8x16(Vector128.ShiftRightLogical(a0, 4) | hh0, d2, -mm2, dst + 32);
                Simd.StoreU8x16(Vector128.ShiftRightLogical(a1, 4) | hh1, d2, -mm2, dst + 48);
            }
            dst -= 256;
        }
    }

    static void Q6_K(byte* p, long nb, float* dst)
    {
        var mask = Vector128.Create((byte)0x0F);
        var m3 = Vector128.Create((byte)3);
        for (long b = 0; b < nb; b++, p += 210, dst += 256)
        {
            byte* ql = p;
            byte* qh = p + 128;
            sbyte* sc = (sbyte*)(p + 192);
            float d = H(p + 208);
            for (int n = 0; n < 2; n++, ql += 64, qh += 32, sc += 8, dst += 128)
            {
                for (int h = 0; h < 2; h++)
                {
                    var qa = Vector128.Load(ql + 16 * h);
                    var qb = Vector128.Load(ql + 32 + 16 * h);
                    var qq = Vector128.Load(qh + 16 * h);
                    var v1 = (qa & mask) | Vector128.ShiftLeft(qq & m3, 4);
                    var v2 = (qb & mask) | Vector128.ShiftLeft(Vector128.ShiftRightLogical(qq, 2) & m3, 4);
                    var v3 = Vector128.ShiftRightLogical(qa, 4) | Vector128.ShiftLeft(Vector128.ShiftRightLogical(qq, 4) & m3, 4);
                    var v4 = Vector128.ShiftRightLogical(qb, 4) | Vector128.ShiftLeft(Vector128.ShiftRightLogical(qq, 6) & m3, 4);
                    float s1 = d * sc[h], s2 = d * sc[h + 2], s3 = d * sc[h + 4], s4 = d * sc[h + 6];
                    Simd.StoreU8x16(v1, s1, -32f * s1, dst + 16 * h);
                    Simd.StoreU8x16(v2, s2, -32f * s2, dst + 32 + 16 * h);
                    Simd.StoreU8x16(v3, s3, -32f * s3, dst + 64 + 16 * h);
                    Simd.StoreU8x16(v4, s4, -32f * s4, dst + 96 + 16 * h);
                }
            }
            dst -= 256;
        }
    }

    // ---- fused matrix-vector kernels (single token): dequantize in registers, no scratch buffer ------------

    public static bool HasFusedKernel(WeightFormat f) =>
        System.Runtime.Intrinsics.X86.Avx512F.IsSupported && f is WeightFormat.Q8_0 or WeightFormat.Q4_0;

    /// <summary>Dot products of x[nb*32] with four Q8_0 / Q4_0 rows.</summary>
    public static void DotRows4(WeightFormat f, float* x, byte** rows, long nb, float* res)
    {
        if (f == WeightFormat.Q8_0) DotQ8_0x4(x, rows[0], rows[1], rows[2], rows[3], nb, res);
        else DotQ4_0x4(x, rows[0], rows[1], rows[2], rows[3], nb, res);
    }

    static void DotQ8_0x4(float* x, byte* r0, byte* r1, byte* r2, byte* r3, long nb, float* res)
    {
        var a0 = Vector512<float>.Zero; var a1 = a0; var a2 = a0; var a3 = a0;
        for (long b = 0; b < nb; b++)
        {
            var x0 = Vector512.Load(x + b * 32); var x1 = Vector512.Load(x + b * 32 + 16);
            long o = b * 34;
            a0 = Row8(r0 + o, x0, x1, a0); a1 = Row8(r1 + o, x0, x1, a1); a2 = Row8(r2 + o, x0, x1, a2); a3 = Row8(r3 + o, x0, x1, a3);
        }
        res[0] = Vector512.Sum(a0); res[1] = Vector512.Sum(a1); res[2] = Vector512.Sum(a2); res[3] = Vector512.Sum(a3);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector512<float> Row8(byte* p, Vector512<float> x0, Vector512<float> x1, Vector512<float> acc)
    {
        var q0 = Vector512.ConvertToSingle(System.Runtime.Intrinsics.X86.Avx512F.ConvertToVector512Int32(Vector128.Load((sbyte*)(p + 2))));
        var q1 = Vector512.ConvertToSingle(System.Runtime.Intrinsics.X86.Avx512F.ConvertToVector512Int32(Vector128.Load((sbyte*)(p + 18))));
        var s = Vector512.FusedMultiplyAdd(q0, x0, q1 * x1);
        return Vector512.FusedMultiplyAdd(s, Vector512.Create(H(p)), acc);
    }

    static void DotQ4_0x4(float* x, byte* r0, byte* r1, byte* r2, byte* r3, long nb, float* res)
    {
        var a0 = Vector512<float>.Zero; var a1 = a0; var a2 = a0; var a3 = a0;
        var mask = Vector128.Create((byte)0x0F);
        for (long b = 0; b < nb; b++)
        {
            var x0 = Vector512.Load(x + b * 32); var x1 = Vector512.Load(x + b * 32 + 16);
            var x8 = (x0 + x1) * Vector512.Create(8f);
            long o = b * 18;
            a0 = Row4(r0 + o, x0, x1, x8, mask, a0); a1 = Row4(r1 + o, x0, x1, x8, mask, a1);
            a2 = Row4(r2 + o, x0, x1, x8, mask, a2); a3 = Row4(r3 + o, x0, x1, x8, mask, a3);
        }
        res[0] = Vector512.Sum(a0); res[1] = Vector512.Sum(a1); res[2] = Vector512.Sum(a2); res[3] = Vector512.Sum(a3);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector512<float> Row4(byte* p, Vector512<float> x0, Vector512<float> x1, Vector512<float> x8, Vector128<byte> mask, Vector512<float> acc)
    {
        var q = Vector128.Load(p + 2);
        var lo = Vector512.ConvertToSingle(System.Runtime.Intrinsics.X86.Avx512F.ConvertToVector512Int32(q & mask));
        var hi = Vector512.ConvertToSingle(System.Runtime.Intrinsics.X86.Avx512F.ConvertToVector512Int32(Vector128.ShiftRightLogical(q, 4)));
        var s = Vector512.FusedMultiplyAdd(lo, x0, hi * x1) - x8;
        return Vector512.FusedMultiplyAdd(s, Vector512.Create(H(p)), acc);
    }
}
