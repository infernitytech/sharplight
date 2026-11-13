using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace Sharplight.Cpu;

// ggml's grid based IQ formats (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S, IQ1_M). A block stores indices into a fixed lattice
// grid (tables in Grids/), sign patterns and small scales. The grids are expanded to floats once so a group of eight values is one
// vector load, one sign-vector load and one multiply.
internal static unsafe partial class Quant
{
    private const float Iq1Delta = 0.125f;

    private static readonly float[] s_negSigns = BuildNegSigns();
    private static float[]? s_iq2XxsFloats;
    private static float[]? s_iq2XsFloats;
    private static float[]? s_iq2SFloats;
    private static float[]? s_iq3XxsFloats;
    private static float[]? s_iq3SFloats;
    private static float[]? s_iq1Floats;

    /// <summary>[256][8]: -1 where bit j of the index is set, +1 elsewhere (ggml's kmask_iq2xs test).</summary>
    private static float[] BuildNegSigns()
    {
        var table = new float[256 * 8];
        for (int b = 0; b < 256; b++)
        {
            for (int j = 0; j < 8; j++)
            {
                table[b * 8 + j] = ((b >> j) & 1) != 0 ? -1f : 1f;
            }
        }

        return table;
    }

    private static float[] ExpandUnsigned(ReadOnlySpan<ulong> grid)
    {
        var floats = new float[grid.Length * 8];
        for (int i = 0; i < grid.Length; i++)
        {
            for (int j = 0; j < 8; j++)
            {
                floats[i * 8 + j] = (byte)(grid[i] >> (8 * j));
            }
        }

        return floats;
    }

    private static float[] ExpandSigned(ReadOnlySpan<ulong> grid)
    {
        var floats = new float[grid.Length * 8];
        for (int i = 0; i < grid.Length; i++)
        {
            for (int j = 0; j < 8; j++)
            {
                floats[i * 8 + j] = (sbyte)(grid[i] >> (8 * j));
            }
        }

        return floats;
    }

    private static float[] ExpandUnsigned(ReadOnlySpan<uint> grid)
    {
        var floats = new float[grid.Length * 4];
        for (int i = 0; i < grid.Length; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                floats[i * 4 + j] = (byte)(grid[i] >> (8 * j));
            }
        }

        return floats;
    }

    private static float[] Iq2XxsFloats() => s_iq2XxsFloats ??= ExpandUnsigned(Iq2XxsGrid.Values);

    private static float[] Iq2XsFloats() => s_iq2XsFloats ??= ExpandUnsigned(Iq2XsGrid.Values);

    private static float[] Iq2SFloats() => s_iq2SFloats ??= ExpandUnsigned(Iq2SGrid.Values);

    private static float[] Iq3XxsFloats() => s_iq3XxsFloats ??= ExpandUnsigned(Iq3XxsGrid.Values);

    private static float[] Iq3SFloats() => s_iq3SFloats ??= ExpandUnsigned(Iq3SGrid.Values);

    private static float[] Iq1Floats() => s_iq1Floats ??= ExpandSigned(Iq1SGrid.Values);

    /// <summary>dst[0..8) = grid[index] * sign pattern * scale.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StoreSigned8(float* grid, int index, float* negSigns, int signs, float scale, float* dst)
    {
        (Vector256.Load(grid + index * 8) * Vector256.Load(negSigns + signs * 8) * Vector256.Create(scale)).Store(dst);
    }

    /// <summary>dst[0..8) = (four grid values of index lo, four of index hi) * sign pattern * scale.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StoreSigned4x2(float* grid, int lo, int hi, float* negSigns, int signs, float scale, float* dst)
    {
        var values = Vector256.Create(Vector128.Load(grid + lo * 4), Vector128.Load(grid + hi * 4));
        (values * Vector256.Load(negSigns + signs * 8) * Vector256.Create(scale)).Store(dst);
    }

    /// <summary>IQ2_XXS: 256 values = { f16 d; u16 qs[32] } (66 bytes). Per 32 values two words: four 8-bit grid indices, then four 7-bit sign indices and a 4-bit scale.</summary>
    private static void IQ2_XXS(byte* p, long nb, float* dst)
    {
        fixed (float* grid = Iq2XxsFloats())
        fixed (float* neg = s_negSigns)
        fixed (byte* ksigns = IqSigns.Values)
        {
            for (long b = 0; b < nb; b++, p += 66)
            {
                float d = H(p);
                for (int ib32 = 0; ib32 < 8; ib32++, dst += 32)
                {
                    uint indices = Unsafe.ReadUnaligned<uint>(p + 2 + 8 * ib32);
                    uint signsAndScale = Unsafe.ReadUnaligned<uint>(p + 6 + 8 * ib32);
                    float db = d * (0.5f + (signsAndScale >> 28)) * 0.25f;
                    for (int l = 0; l < 4; l++)
                    {
                        int signs = ksigns[(signsAndScale >> (7 * l)) & 127];
                        StoreSigned8(grid, (int)((indices >> (8 * l)) & 0xFF), neg, signs, db, dst + 8 * l);
                    }
                }
            }
        }
    }

    /// <summary>IQ2_XS: 256 values = { f16 d; u16 qs[32]; u8 scales[8] } (74 bytes). Each u16 holds a 9-bit grid index and a 7-bit sign index; scales are two nibbles per 32 values.</summary>
    private static void IQ2_XS(byte* p, long nb, float* dst)
    {
        fixed (float* grid = Iq2XsFloats())
        fixed (float* neg = s_negSigns)
        fixed (byte* ksigns = IqSigns.Values)
        {
            for (long b = 0; b < nb; b++, p += 74)
            {
                float d = H(p);
                for (int ib32 = 0; ib32 < 8; ib32++, dst += 32)
                {
                    int scales = p[66 + ib32];
                    float db0 = d * (0.5f + (scales & 0xF)) * 0.25f;
                    float db1 = d * (0.5f + (scales >> 4)) * 0.25f;
                    for (int l = 0; l < 4; l++)
                    {
                        int q = Unsafe.ReadUnaligned<ushort>(p + 2 + 2 * (4 * ib32 + l));
                        StoreSigned8(grid, q & 511, neg, ksigns[q >> 9], l < 2 ? db0 : db1, dst + 8 * l);
                    }
                }
            }
        }
    }

    /// <summary>IQ2_S: 256 values = { f16 d; u8 qs[64]; u8 qh[8]; u8 scales[8] } (82 bytes). qs[0..32) are the low grid index bits, qs[32..64) the sign bytes, qh the two high index bits per group of 8.</summary>
    private static void IQ2_S(byte* p, long nb, float* dst)
    {
        fixed (float* grid = Iq2SFloats())
        fixed (float* neg = s_negSigns)
        {
            for (long b = 0; b < nb; b++, p += 82)
            {
                float d = H(p);
                byte* qs = p + 2;
                byte* qh = p + 66;
                byte* signs = qs + 32;
                for (int ib32 = 0; ib32 < 8; ib32++, dst += 32, qs += 4, signs += 4)
                {
                    int scales = p[74 + ib32];
                    float db0 = d * (0.5f + (scales & 0xF)) * 0.25f;
                    float db1 = d * (0.5f + (scales >> 4)) * 0.25f;
                    for (int l = 0; l < 4; l++)
                    {
                        int index = qs[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300);
                        StoreSigned8(grid, index, neg, signs[l], l < 2 ? db0 : db1, dst + 8 * l);
                    }
                }
            }
        }
    }

    /// <summary>IQ3_XXS: 256 values = { f16 d; u8 qs[96] } (98 bytes). qs[0..64) are 8-bit indices of four-value grid entries, qs[64..96) one word per 32 values: four 7-bit sign indices and a 4-bit scale.</summary>
    private static void IQ3_XXS(byte* p, long nb, float* dst)
    {
        fixed (float* grid = Iq3XxsFloats())
        fixed (float* neg = s_negSigns)
        fixed (byte* ksigns = IqSigns.Values)
        {
            for (long b = 0; b < nb; b++, p += 98)
            {
                float d = H(p);
                byte* qs = p + 2;
                byte* scalesAndSigns = qs + 64;
                for (int ib32 = 0; ib32 < 8; ib32++, dst += 32, qs += 8)
                {
                    uint word = Unsafe.ReadUnaligned<uint>(scalesAndSigns + 4 * ib32);
                    float db = d * (0.5f + (word >> 28)) * 0.5f;
                    for (int l = 0; l < 4; l++)
                    {
                        int signs = ksigns[(word >> (7 * l)) & 127];
                        StoreSigned4x2(grid, qs[2 * l], qs[2 * l + 1], neg, signs, db, dst + 8 * l);
                    }
                }
            }
        }
    }

    /// <summary>IQ3_S: 256 values = { f16 d; u8 qs[64]; u8 qh[8]; u8 signs[32]; u8 scales[4] } (110 bytes). Nine-bit indices of four-value grid entries (the ninth bit in qh), explicit sign bytes, odd 4-bit scales (1 + 2 s) per 32 values.</summary>
    private static void IQ3_S(byte* p, long nb, float* dst)
    {
        fixed (float* grid = Iq3SFloats())
        fixed (float* neg = s_negSigns)
        {
            for (long b = 0; b < nb; b++, p += 110)
            {
                float d = H(p);
                byte* qs = p + 2;
                byte* qh = p + 66;
                byte* signs = p + 74;
                for (int ib32 = 0; ib32 < 8; ib32 += 2)
                {
                    float db1 = d * (1 + 2 * (p[106 + ib32 / 2] & 0xF));
                    float db2 = d * (1 + 2 * (p[106 + ib32 / 2] >> 4));
                    for (int half = 0; half < 2; half++, dst += 32, qs += 8, signs += 4)
                    {
                        int high = qh[half];
                        float db = half == 0 ? db1 : db2;
                        for (int l = 0; l < 4; l++)
                        {
                            int lo = qs[2 * l] | ((high << (8 - 2 * l)) & 256);
                            int hi = qs[2 * l + 1] | ((high << (7 - 2 * l)) & 256);
                            StoreSigned4x2(grid, lo, hi, neg, signs[l], db, dst + 8 * l);
                        }
                    }

                    qh += 2;
                }
            }
        }
    }

    /// <summary>dst[0..8) = (grid[index] + delta) * scale.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StoreIq1(float* grid, int index, float delta, float scale, float* dst)
    {
        ((Vector256.Load(grid + index * 8) + Vector256.Create(delta)) * Vector256.Create(scale)).Store(dst);
    }

    /// <summary>IQ1_S: 256 values = { f16 d; u8 qs[32]; u16 qh[8] } (50 bytes). Per 32 values a u16 of qh: three high index bits for each of four groups, a 3-bit scale (2s+1) and a delta sign.</summary>
    private static void IQ1_S(byte* p, long nb, float* dst)
    {
        fixed (float* grid = Iq1Floats())
        {
            for (long b = 0; b < nb; b++, p += 50)
            {
                float d = H(p);
                byte* qs = p + 2;
                for (int ib = 0; ib < 8; ib++, dst += 32, qs += 4)
                {
                    int qh = Unsafe.ReadUnaligned<ushort>(p + 34 + 2 * ib);
                    float dl = d * (2 * ((qh >> 12) & 7) + 1);
                    float delta = (qh & 0x8000) != 0 ? -Iq1Delta : Iq1Delta;
                    for (int l = 0; l < 4; l++)
                    {
                        StoreIq1(grid, qs[l] | (((qh >> (3 * l)) & 7) << 8), delta, dl, dst + 8 * l);
                    }
                }
            }
        }
    }

    /// <summary>IQ1_M: 256 values = { u8 qs[32]; u8 qh[16]; u16 scales[4] } (56 bytes). Like IQ1_S with a 3-bit scale per 16 values and a delta sign per 8; the f16 super-scale is spread over the top nibbles of the four scale words.</summary>
    private static void IQ1_M(byte* p, long nb, float* dst)
    {
        fixed (float* grid = Iq1Floats())
        {
            for (long b = 0; b < nb; b++, p += 56)
            {
                ushort* sc = (ushort*)(p + 48);
                int scaleBits = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
                float d = Simd.HalfToFloat((ushort)scaleBits);
                byte* qs = p;
                byte* qh = p + 32;
                for (int ib = 0; ib < 8; ib++, dst += 32, qs += 4, qh += 2)
                {
                    float dl1 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 7) + 1);
                    float dl2 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 7) + 1);
                    StoreIq1(grid, qs[0] | ((qh[0] << 8) & 0x700), (qh[0] & 0x08) != 0 ? -Iq1Delta : Iq1Delta, dl1, dst);
                    StoreIq1(grid, qs[1] | ((qh[0] << 4) & 0x700), (qh[0] & 0x80) != 0 ? -Iq1Delta : Iq1Delta, dl1, dst + 8);
                    StoreIq1(grid, qs[2] | ((qh[1] << 8) & 0x700), (qh[1] & 0x08) != 0 ? -Iq1Delta : Iq1Delta, dl2, dst + 16);
                    StoreIq1(grid, qs[3] | ((qh[1] << 4) & 0x700), (qh[1] & 0x80) != 0 ? -Iq1Delta : Iq1Delta, dl2, dst + 24);
                }
            }
        }
    }
}
