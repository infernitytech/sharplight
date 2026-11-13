using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace Sharplight.Cpu;

// Weight row decoders: ggml block formats -> unsigned codes u plus a multiplier a and an offset b per 16 values (value = a * u + b).
internal static unsafe partial class IntDot
{
    /// <summary>The IQ4_NL / IQ4_XS codebook shifted by 128 into unsigned codes.</summary>
    private static readonly Vector128<byte> s_iq4Codes = Vector128.Create(
        (byte)1, 24, 45, 63, 79, 93, 106, 118, 129, 141, 153, 166, 181, 197, 217, 241);

    /// <summary>The MXFP4 doubled E2M1 values shifted by 16 into unsigned codes.</summary>
    private static readonly Vector128<byte> s_fp4Codes = Vector128.Create(
        (byte)16, 17, 18, 19, 20, 22, 24, 28, 16, 15, 14, 13, 12, 10, 8, 4);

    /// <summary>Decodes <paramref name="k"/> values of a weight row in format <paramref name="f"/> (<see cref="Supports"/>).</summary>
    public static void Decode(WeightFormat f, byte* src, long k, byte* u, float* a, float* b)
    {
        long nb = k / Quant.BlockSize(f);
        switch (f)
        {
            case WeightFormat.Q8_0:
                DecodeQ8_0(src, nb, u, a, b);
                break;
            case WeightFormat.Q4_0:
                DecodeQ4(src, nb, u, a, b, hasMin: false);
                break;
            case WeightFormat.Q4_1:
                DecodeQ4(src, nb, u, a, b, hasMin: true);
                break;
            case WeightFormat.Q5_0:
                DecodeQ5(src, nb, u, a, b, hasMin: false);
                break;
            case WeightFormat.Q5_1:
                DecodeQ5(src, nb, u, a, b, hasMin: true);
                break;
            case WeightFormat.Q4_K:
                DecodeQ4_K(src, nb, u, a, b);
                break;
            case WeightFormat.Q5_K:
                DecodeQ5_K(src, nb, u, a, b);
                break;
            case WeightFormat.Q6_K:
                DecodeQ6_K(src, nb, u, a, b);
                break;
            case WeightFormat.IQ4_NL:
                DecodeIq4Nl(src, nb, u, a, b);
                break;
            case WeightFormat.IQ4_XS:
                DecodeIq4Xs(src, nb, u, a, b);
                break;
            case WeightFormat.MXFP4:
                DecodeMxfp4(src, nb, u, a, b);
                break;
            default:
                throw new NotSupportedException($"{f} has no integer-dot decoder.");
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void SetBlock(float* a, float* b, float multiplier, float offset)
    {
        a[0] = multiplier;
        a[1] = multiplier;
        b[0] = offset;
        b[1] = offset;
    }

    /// <summary>Q8_0 { f16 d; i8 q[32] }: u = q + 128, value = d * u - 128 * d.</summary>
    private static void DecodeQ8_0(byte* p, long nb, byte* u, float* a, float* b)
    {
        var flip = Vector128.Create((byte)0x80);
        for (long blk = 0; blk < nb; blk++, p += 34, u += 32, a += 2, b += 2)
        {
            float d = Quant.H(p);
            (Vector128.Load(p + 2) ^ flip).Store(u);
            (Vector128.Load(p + 18) ^ flip).Store(u + 16);
            SetBlock(a, b, d, -128f * d);
        }
    }

    /// <summary>Q4_0 { f16 d; u8 qs[16] } (value d * (q - 8)) and Q4_1 { f16 d; f16 m; u8 qs[16] } (value d * q + m).</summary>
    private static void DecodeQ4(byte* p, long nb, byte* u, float* a, float* b, bool hasMin)
    {
        var mask = Vector128.Create((byte)0x0F);
        int bytes = hasMin ? 20 : 18;
        int qOffset = hasMin ? 4 : 2;
        for (long blk = 0; blk < nb; blk++, p += bytes, u += 32, a += 2, b += 2)
        {
            float d = Quant.H(p);
            var q = Vector128.Load(p + qOffset);
            (q & mask).Store(u);
            Vector128.ShiftRightLogical(q, 4).Store(u + 16);
            SetBlock(a, b, d, hasMin ? Quant.H(p + 2) : -8f * d);
        }
    }

    /// <summary>Q5_0 { f16 d; u32 qh; u8 qs[16] } (value d * (q - 16)) and Q5_1 { f16 d; f16 m; u32 qh; u8 qs[16] } (value d * q + m).</summary>
    private static void DecodeQ5(byte* p, long nb, byte* u, float* a, float* b, bool hasMin)
    {
        var mask = Vector128.Create((byte)0x0F);
        int bytes = hasMin ? 24 : 22;
        int hOffset = hasMin ? 4 : 2;
        byte* high = stackalloc byte[32];
        for (long blk = 0; blk < nb; blk++, p += bytes, u += 32, a += 2, b += 2)
        {
            float d = Quant.H(p);
            uint qh = Unsafe.ReadUnaligned<uint>(p + hOffset);
            for (int j = 0; j < 32; j++)
            {
                high[j] = (byte)(((qh >> j) & 1) << 4);
            }

            var q = Vector128.Load(p + hOffset + 4);
            ((q & mask) | Vector128.Load(high)).Store(u);
            (Vector128.ShiftRightLogical(q, 4) | Vector128.Load(high + 16)).Store(u + 16);
            SetBlock(a, b, d, hasMin ? Quant.H(p + 2) : -16f * d);
        }
    }

    /// <summary>Q4_K { f16 d; f16 dmin; u8 scales[12]; u8 qs[128] }: eight 32-value sub-blocks, value d * sc * q - dmin * m.</summary>
    private static void DecodeQ4_K(byte* p, long nb, byte* u, float* a, float* b)
    {
        var mask = Vector128.Create((byte)0x0F);
        for (long blk = 0; blk < nb; blk++, p += 144, u += 256, a += 16, b += 16)
        {
            float d = Quant.H(p);
            float dmin = Quant.H(p + 2);
            byte* qs = p + 16;
            for (int j = 0; j < 4; j++)
            {
                var q0 = Vector128.Load(qs + 32 * j);
                var q1 = Vector128.Load(qs + 32 * j + 16);
                byte* uj = u + 64 * j;
                (q0 & mask).Store(uj);
                (q1 & mask).Store(uj + 16);
                Vector128.ShiftRightLogical(q0, 4).Store(uj + 32);
                Vector128.ShiftRightLogical(q1, 4).Store(uj + 48);
            }

            SubBlockScales(p + 4, d, dmin, a, b);
        }
    }

    /// <summary>Q5_K { f16 d; f16 dmin; u8 scales[12]; u8 qh[32]; u8 qs[128] }: like Q4_K with a fifth bit from qh.</summary>
    private static void DecodeQ5_K(byte* p, long nb, byte* u, float* a, float* b)
    {
        var mask = Vector128.Create((byte)0x0F);
        var one = Vector128.Create((byte)1);
        for (long blk = 0; blk < nb; blk++, p += 176, u += 256, a += 16, b += 16)
        {
            float d = Quant.H(p);
            float dmin = Quant.H(p + 2);
            var h0 = Vector128.Load(p + 16);
            var h1 = Vector128.Load(p + 32);
            byte* qs = p + 48;
            for (int j = 0; j < 4; j++)
            {
                var q0 = Vector128.Load(qs + 32 * j);
                var q1 = Vector128.Load(qs + 32 * j + 16);
                byte* uj = u + 64 * j;
                ((q0 & mask) | Vector128.ShiftLeft(Vector128.ShiftRightLogical(h0, 2 * j) & one, 4)).Store(uj);
                ((q1 & mask) | Vector128.ShiftLeft(Vector128.ShiftRightLogical(h1, 2 * j) & one, 4)).Store(uj + 16);
                (Vector128.ShiftRightLogical(q0, 4) | Vector128.ShiftLeft(Vector128.ShiftRightLogical(h0, 2 * j + 1) & one, 4)).Store(uj + 32);
                (Vector128.ShiftRightLogical(q1, 4) | Vector128.ShiftLeft(Vector128.ShiftRightLogical(h1, 2 * j + 1) & one, 4)).Store(uj + 48);
            }

            SubBlockScales(p + 4, d, dmin, a, b);
        }
    }

    /// <summary>The 6-bit scales and mins of the eight 32-value sub-blocks of Q4_K / Q5_K, two 16-value groups each.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void SubBlockScales(byte* scales, float d, float dmin, float* a, float* b)
    {
        for (int j = 0; j < 8; j++)
        {
            Quant.ScaleMin(j, scales, out int sc, out int m);
            SetBlock(a + 2 * j, b + 2 * j, d * sc, -dmin * m);
        }
    }

    /// <summary>Q6_K { u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d }: value d * sc * (q - 32), one scale per 16 values in order.</summary>
    private static void DecodeQ6_K(byte* p, long nb, byte* u, float* a, float* b)
    {
        var mask = Vector128.Create((byte)0x0F);
        var m3 = Vector128.Create((byte)3);
        for (long blk = 0; blk < nb; blk++, p += 210, u += 256, a += 16, b += 16)
        {
            float d = Quant.H(p + 208);
            for (int n = 0; n < 2; n++)
            {
                byte* ql = p + 64 * n;
                byte* qh = p + 128 + 32 * n;
                byte* un = u + 128 * n;
                for (int h = 0; h < 2; h++)
                {
                    var qa = Vector128.Load(ql + 16 * h);
                    var qb = Vector128.Load(ql + 32 + 16 * h);
                    var qq = Vector128.Load(qh + 16 * h);
                    ((qa & mask) | Vector128.ShiftLeft(qq & m3, 4)).Store(un + 16 * h);
                    ((qb & mask) | Vector128.ShiftLeft(Vector128.ShiftRightLogical(qq, 2) & m3, 4)).Store(un + 32 + 16 * h);
                    (Vector128.ShiftRightLogical(qa, 4) | Vector128.ShiftLeft(Vector128.ShiftRightLogical(qq, 4) & m3, 4)).Store(un + 64 + 16 * h);
                    (Vector128.ShiftRightLogical(qb, 4) | Vector128.ShiftLeft(Vector128.ShiftRightLogical(qq, 6) & m3, 4)).Store(un + 96 + 16 * h);
                }
            }

            sbyte* sc = (sbyte*)(p + 192);
            for (int g = 0; g < 16; g++)
            {
                float s = d * sc[g];
                a[g] = s;
                b[g] = -32f * s;
            }
        }
    }

    /// <summary>IQ4_NL { f16 d; u8 qs[16] }: a nibble indexes the non-linear codebook, value d * kvalues[q].</summary>
    private static void DecodeIq4Nl(byte* p, long nb, byte* u, float* a, float* b)
    {
        var mask = Vector128.Create((byte)0x0F);
        for (long blk = 0; blk < nb; blk++, p += 18, u += 32, a += 2, b += 2)
        {
            float d = Quant.H(p);
            var q = Vector128.Load(p + 2);
            Vector128.Shuffle(s_iq4Codes, q & mask).Store(u);
            Vector128.Shuffle(s_iq4Codes, Vector128.ShiftRightLogical(q, 4)).Store(u + 16);
            SetBlock(a, b, d, -128f * d);
        }
    }

    /// <summary>IQ4_XS { f16 d; u16 scales_h; u8 scales_l[4]; u8 qs[128] }: eight IQ4_NL-like sub-blocks with 6-bit scales (ls - 32).</summary>
    private static void DecodeIq4Xs(byte* p, long nb, byte* u, float* a, float* b)
    {
        var mask = Vector128.Create((byte)0x0F);
        for (long blk = 0; blk < nb; blk++, p += 136, u += 256, a += 16, b += 16)
        {
            float d = Quant.H(p);
            uint sh = p[2] | (uint)p[3] << 8;
            for (int ib = 0; ib < 8; ib++)
            {
                int ls = ((p[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) | (int)(((sh >> (2 * ib)) & 3) << 4);
                float dl = d * (ls - 32);
                var q = Vector128.Load(p + 8 + 16 * ib);
                Vector128.Shuffle(s_iq4Codes, q & mask).Store(u + 32 * ib);
                Vector128.Shuffle(s_iq4Codes, Vector128.ShiftRightLogical(q, 4)).Store(u + 32 * ib + 16);
                SetBlock(a + 2 * ib, b + 2 * ib, dl, -128f * dl);
            }
        }
    }

    /// <summary>MXFP4 { u8 e; u8 qs[16] }: a nibble indexes the doubled E2M1 values, value fp4[q] * 2^(e-127) / 2.</summary>
    private static void DecodeMxfp4(byte* p, long nb, byte* u, float* a, float* b)
    {
        var mask = Vector128.Create((byte)0x0F);
        for (long blk = 0; blk < nb; blk++, p += 17, u += 32, a += 2, b += 2)
        {
            float d = Quant.E8M0Half(p[0]);
            var q = Vector128.Load(p + 1);
            Vector128.Shuffle(s_fp4Codes, q & mask).Store(u);
            Vector128.Shuffle(s_fp4Codes, Vector128.ShiftRightLogical(q, 4)).Store(u + 16);
            SetBlock(a, b, d, -16f * d);
        }
    }
}
