using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using Sharplight.Tensors.Grids;

namespace Sharplight.Cpu;

// Integer-dot decoders of ggml's grid formats IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS and IQ3_S. A grid entry holds unsigned magnitudes below 64
// (at most 43 for the IQ2 grids, 62 for IQ3_XXS, 15 for IQ3_S) and a sign pattern negates some of them, so a value is
// scale * (u - 64) with the code u = 64 ± magnitude in 1..127: the small-code dot applies.
internal static unsafe partial class IntDot
{
    /// <summary>The code of a grid value of zero.</summary>
    private const int IqZero = 64;

    /// <summary>[256]: byte j is 0xFF where bit j of the sign pattern is set (the value is negated).</summary>
    private static readonly ulong[] s_iqSignMasks = BuildIqSignMasks();

    private static ulong[] BuildIqSignMasks()
    {
        var masks = new ulong[256];
        for (int s = 0; s < 256; s++)
        {
            for (int j = 0; j < 8; j++)
            {
                if (((s >> j) & 1) != 0)
                {
                    masks[s] |= 0xFFUL << (8 * j);
                }
            }
        }

        return masks;
    }

    /// <summary>Codes of two groups of eight grid values: 64 + magnitude, or 64 - magnitude where the sign mask is set.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StoreIqCodes(ulong grid0, ulong mask0, ulong grid1, ulong mask1, byte* u)
    {
        var g = Vector128.Create(grid0, grid1).AsByte();
        var m = Vector128.Create(mask0, mask1).AsByte();
        ((g ^ m) - m + Vector128.Create((byte)IqZero)).Store(u);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void SetIqScales(float* a, float* b, float first, float second)
    {
        a[0] = first;
        a[1] = second;
        b[0] = -IqZero * first;
        b[1] = -IqZero * second;
    }

    /// <summary>IQ2_XXS { f16 d; u16 qs[32] }: per 32 values four 8-bit grid indices, four 7-bit sign indices and a 4-bit scale.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void DecodeIq2Xxs(byte* p, long nb, byte* u, float* a, float* b)
    {
        fixed (ulong* grid = Iq2XxsGrid.Values)
        fixed (byte* ksigns = IqSigns.Values)
        fixed (ulong* masks = s_iqSignMasks)
        {
            for (long blk = 0; blk < nb; blk++, p += 66)
            {
                float d = Quant.H(p);
                for (int ib32 = 0; ib32 < 8; ib32++, u += 32, a += 2, b += 2)
                {
                    uint indices = Unsafe.ReadUnaligned<uint>(p + 2 + 8 * ib32);
                    uint signsAndScale = Unsafe.ReadUnaligned<uint>(p + 6 + 8 * ib32);
                    float db = d * (0.5f + (signsAndScale >> 28)) * 0.25f;
                    for (int l = 0; l < 4; l += 2)
                    {
                        StoreIqCodes(
                            grid[(indices >> (8 * l)) & 0xFF], masks[ksigns[(signsAndScale >> (7 * l)) & 127]],
                            grid[(indices >> (8 * l + 8)) & 0xFF], masks[ksigns[(signsAndScale >> (7 * l + 7)) & 127]], u + 8 * l);
                    }

                    SetIqScales(a, b, db, db);
                }
            }
        }
    }

    /// <summary>IQ2_XS { f16 d; u16 qs[32]; u8 scales[8] }: each u16 a 9-bit grid index and a 7-bit sign index; a 4-bit scale per 16 values.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void DecodeIq2Xs(byte* p, long nb, byte* u, float* a, float* b)
    {
        fixed (ulong* grid = Iq2XsGrid.Values)
        fixed (byte* ksigns = IqSigns.Values)
        fixed (ulong* masks = s_iqSignMasks)
        {
            for (long blk = 0; blk < nb; blk++, p += 74)
            {
                float d = Quant.H(p);
                ushort* qs = (ushort*)(p + 2);
                for (int ib32 = 0; ib32 < 8; ib32++, u += 32, a += 2, b += 2, qs += 4)
                {
                    int scales = p[66 + ib32];
                    for (int l = 0; l < 4; l += 2)
                    {
                        int q0 = Unsafe.ReadUnaligned<ushort>(qs + l);
                        int q1 = Unsafe.ReadUnaligned<ushort>(qs + l + 1);
                        StoreIqCodes(grid[q0 & 511], masks[ksigns[q0 >> 9]], grid[q1 & 511], masks[ksigns[q1 >> 9]], u + 8 * l);
                    }

                    SetIqScales(a, b, d * (0.5f + (scales & 0xF)) * 0.25f, d * (0.5f + (scales >> 4)) * 0.25f);
                }
            }
        }
    }

    /// <summary>IQ2_S { f16 d; u8 qs[64]; u8 qh[8]; u8 scales[8] }: 10-bit grid indices (two high bits in qh), explicit sign bytes, a 4-bit scale per 16 values.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void DecodeIq2S(byte* p, long nb, byte* u, float* a, float* b)
    {
        fixed (ulong* grid = Iq2SGrid.Values)
        fixed (ulong* masks = s_iqSignMasks)
        {
            for (long blk = 0; blk < nb; blk++, p += 82)
            {
                float d = Quant.H(p);
                byte* qs = p + 2;
                byte* signs = qs + 32;
                byte* qh = p + 66;
                for (int ib32 = 0; ib32 < 8; ib32++, u += 32, a += 2, b += 2, qs += 4, signs += 4)
                {
                    int scales = p[74 + ib32];
                    int high = qh[ib32];
                    for (int l = 0; l < 4; l += 2)
                    {
                        int i0 = qs[l] | ((high << (8 - 2 * l)) & 0x300);
                        int i1 = qs[l + 1] | ((high << (6 - 2 * l)) & 0x300);
                        StoreIqCodes(grid[i0], masks[signs[l]], grid[i1], masks[signs[l + 1]], u + 8 * l);
                    }

                    SetIqScales(a, b, d * (0.5f + (scales & 0xF)) * 0.25f, d * (0.5f + (scales >> 4)) * 0.25f);
                }
            }
        }
    }

    /// <summary>IQ3_XXS { f16 d; u8 qs[96] }: 8-bit indices of four-value grid entries, then per 32 values four 7-bit sign indices and a 4-bit scale.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void DecodeIq3Xxs(byte* p, long nb, byte* u, float* a, float* b)
    {
        fixed (uint* grid = Iq3XxsGrid.Values)
        fixed (byte* ksigns = IqSigns.Values)
        fixed (ulong* masks = s_iqSignMasks)
        {
            for (long blk = 0; blk < nb; blk++, p += 98)
            {
                float d = Quant.H(p);
                byte* qs = p + 2;
                byte* scalesAndSigns = qs + 64;
                for (int ib32 = 0; ib32 < 8; ib32++, u += 32, a += 2, b += 2, qs += 8)
                {
                    uint word = Unsafe.ReadUnaligned<uint>(scalesAndSigns + 4 * ib32);
                    float db = d * (0.5f + (word >> 28)) * 0.5f;
                    for (int l = 0; l < 4; l += 2)
                    {
                        ulong g0 = grid[qs[2 * l]] | ((ulong)grid[qs[2 * l + 1]] << 32);
                        ulong g1 = grid[qs[2 * l + 2]] | ((ulong)grid[qs[2 * l + 3]] << 32);
                        StoreIqCodes(g0, masks[ksigns[(word >> (7 * l)) & 127]], g1, masks[ksigns[(word >> (7 * l + 7)) & 127]], u + 8 * l);
                    }

                    SetIqScales(a, b, db, db);
                }
            }
        }
    }

    /// <summary>IQ3_S { f16 d; u8 qs[64]; u8 qh[8]; u8 signs[32]; u8 scales[4] }: 9-bit indices of four-value grid entries, explicit sign bytes, odd 4-bit scales per 32 values.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void DecodeIq3S(byte* p, long nb, byte* u, float* a, float* b)
    {
        fixed (uint* grid = Iq3SGrid.Values)
        fixed (ulong* masks = s_iqSignMasks)
        {
            for (long blk = 0; blk < nb; blk++, p += 110)
            {
                float d = Quant.H(p);
                byte* qs = p + 2;
                byte* qh = p + 66;
                byte* signs = p + 74;
                for (int ib32 = 0; ib32 < 8; ib32++, u += 32, a += 2, b += 2, qs += 8, signs += 4)
                {
                    int scale = p[106 + ib32 / 2] >> (4 * (ib32 & 1)) & 0xF;
                    float db = d * (1 + 2 * scale);
                    int high = qh[ib32];
                    for (int l = 0; l < 4; l += 2)
                    {
                        ulong g0 = Iq3SPair(grid, qs, high, l);
                        ulong g1 = Iq3SPair(grid, qs, high, l + 1);
                        StoreIqCodes(g0, masks[signs[l]], g1, masks[signs[l + 1]], u + 8 * l);
                    }

                    SetIqScales(a, b, db, db);
                }
            }
        }
    }

    /// <summary>The eight magnitudes of group <paramref name="l"/> of an IQ3_S sub-block: two grid entries with their ninth index bit from <paramref name="high"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong Iq3SPair(uint* grid, byte* qs, int high, int l)
    {
        int lo = qs[2 * l] | ((high << (8 - 2 * l)) & 256);
        int hi = qs[2 * l + 1] | ((high << (7 - 2 * l)) & 256);
        return grid[lo] | ((ulong)grid[hi] << 32);
    }
}
