using System.Buffers.Binary;
using Sharplight.Tensors;
using Sharplight.Tensors.Grids;

namespace Sharplight.Import.Gguf;

// The 2- and 3-bit k-quants (Q2_K, Q3_K), the lattice formats (IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S), the non-linear
// 4-bit formats (IQ4_NL, IQ4_XS), the ternary formats (TQ1_0, TQ2_0) and the 4-bit floats (MXFP4, NVFP4), one block at a time and with
// ggml's arithmetic order, so the values equal ggml's reference decoders bit for bit.
internal static partial class GgmlBlockDecoder
{
    private const float Iq1Delta = 0.125f;

    private static readonly sbyte[] s_nonLinear4 = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];
    private static readonly sbyte[] s_fp4Doubled = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

    private static bool DecodeLowBit(DataType type, ReadOnlySpan<byte> block, Span<float> y)
    {
        if (type == DataType.Q2_K)
        {
            DecodeQ2K(block, y);
        }
        else if (type == DataType.Q3_K)
        {
            DecodeQ3K(block, y);
        }
        else if (type == DataType.IQ4_NL)
        {
            DecodeNonLinear4(Half16(block), block.Slice(2, 16), y);
        }
        else if (type == DataType.IQ4_XS)
        {
            DecodeIq4Xs(block, y);
        }
        else if (type == DataType.IQ2_XXS)
        {
            DecodeIq2Xxs(block, y);
        }
        else if (type == DataType.IQ2_XS)
        {
            DecodeIq2Xs(block, y);
        }
        else if (type == DataType.IQ2_S)
        {
            DecodeIq2S(block, y);
        }
        else if (type == DataType.IQ3_XXS)
        {
            DecodeIq3Xxs(block, y);
        }
        else if (type == DataType.IQ3_S)
        {
            DecodeIq3S(block, y);
        }
        else if (type == DataType.IQ1_S)
        {
            DecodeIq1S(block, y);
        }
        else if (type == DataType.IQ1_M)
        {
            DecodeIq1M(block, y);
        }
        else if (type == DataType.TQ1_0)
        {
            DecodeTq1(block, y);
        }
        else if (type == DataType.TQ2_0)
        {
            DecodeTq2(block, y);
        }
        else if (type == DataType.MXFP4)
        {
            DecodeFp4(E8M0Half(block[0]), block.Slice(1, 16), y);
        }
        else if (type == DataType.NVFP4)
        {
            for (int s = 0; s < 4; s++)
            {
                DecodeFp4(Ue4M3Half(block[s]), block.Slice(4 + 8 * s, 8), y.Slice(16 * s, 16));
            }
        }
        else
        {
            return false;
        }

        return true;
    }

    // Q2_K: { u8 scales[16]; u8 qs[64]; f16 d; f16 dmin }: sixteen 16-value sub-blocks, value d·scale·q - dmin·min.
    private static void DecodeQ2K(ReadOnlySpan<byte> p, Span<float> y)
    {
        float d = Half16(p[80..]);
        float min = Half16(p[82..]);
        for (int n = 0; n < 2; n++)
        {
            var q = p.Slice(16 + 32 * n, 32);
            for (int j = 0; j < 4; j++)
            {
                for (int h = 0; h < 2; h++)
                {
                    int sc = p[n * 8 + 2 * j + h];
                    float dl = d * (sc & 0xF);
                    float ml = min * (sc >> 4);
                    for (int i = 0; i < 16; i++)
                    {
                        y[128 * n + 32 * j + 16 * h + i] = ((q[16 * h + i] >> (2 * j)) & 3) * dl - ml;
                    }
                }
            }
        }
    }

    // Q3_K: { u8 hmask[32]; u8 qs[64]; u8 scales[12]; f16 d }: 6-bit scales minus 32, the 2-bit code minus 4 where the hmask bit is clear.
    private static void DecodeQ3K(ReadOnlySpan<byte> p, Span<float> y)
    {
        uint a0 = BinaryPrimitives.ReadUInt32LittleEndian(p[96..]);
        uint a1 = BinaryPrimitives.ReadUInt32LittleEndian(p[100..]);
        uint tmp = BinaryPrimitives.ReadUInt32LittleEndian(p[104..]);
        Span<uint> aux =
        [
            (a0 & 0x0F0F0F0F) | ((tmp & 0x03030303) << 4),
            (a1 & 0x0F0F0F0F) | (((tmp >> 2) & 0x03030303) << 4),
            ((a0 >> 4) & 0x0F0F0F0F) | (((tmp >> 4) & 0x03030303) << 4),
            ((a1 >> 4) & 0x0F0F0F0F) | (((tmp >> 6) & 0x03030303) << 4),
        ];
        var scales = System.Runtime.InteropServices.MemoryMarshal.Cast<uint, sbyte>(aux);
        float dAll = Half16(p[108..]);
        for (int n = 0; n < 2; n++)
        {
            var q = p.Slice(32 + 32 * n, 32);
            for (int j = 0; j < 4; j++)
            {
                for (int h = 0; h < 2; h++)
                {
                    float dl = dAll * (scales[n * 8 + 2 * j + h] - 32);
                    for (int i = 0; i < 16; i++)
                    {
                        int low = (q[16 * h + i] >> (2 * j)) & 3;
                        int high = (p[16 * h + i] >> (4 * n + j)) & 1;
                        y[128 * n + 32 * j + 16 * h + i] = (sbyte)(low + (high << 2) - 4) * dl;
                    }
                }
            }
        }
    }

    // 32 values of the non-linear 4-bit grid: j < 16 from the low nibble of qs[j], j >= 16 from the high nibble of qs[j - 16].
    private static void DecodeNonLinear4(float d, ReadOnlySpan<byte> qs, Span<float> y)
    {
        for (int j = 0; j < 16; j++)
        {
            y[j] = d * s_nonLinear4[qs[j] & 0xF];
            y[j + 16] = d * s_nonLinear4[qs[j] >> 4];
        }
    }

    // IQ4_XS: { f16 d; u16 scales_h; u8 scales_l[4]; u8 qs[128] }: eight IQ4_NL-like sub-blocks with 6-bit scales minus 32.
    private static void DecodeIq4Xs(ReadOnlySpan<byte> p, Span<float> y)
    {
        float d = Half16(p);
        uint sh = BinaryPrimitives.ReadUInt16LittleEndian(p[2..]);
        for (int ib = 0; ib < 8; ib++)
        {
            int ls = ((p[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) | (int)(((sh >> (2 * ib)) & 3) << 4);
            DecodeNonLinear4(d * (ls - 32), p.Slice(8 + 16 * ib, 16), y.Slice(32 * ib, 32));
        }
    }

    // Eight values: byte j of a grid entry, negated where bit j of signs is set, times scale.
    private static void StoreSigned(ulong entry, int signs, float scale, Span<float> y)
    {
        for (int j = 0; j < 8; j++)
        {
            y[j] = (byte)(entry >> (8 * j)) * (((signs >> j) & 1) != 0 ? -1f : 1f) * scale;
        }
    }

    // Two four-value grid entries (uint) as one signed group of eight.
    private static ulong Join(uint lo, uint hi) => lo | ((ulong)hi << 32);

    // IQ2_XXS: { f16 d; u16 qs[32] }: per 32 values four 8-bit grid indices, then four 7-bit sign indices and a 4-bit scale.
    private static void DecodeIq2Xxs(ReadOnlySpan<byte> p, Span<float> y)
    {
        float d = Half16(p);
        for (int ib32 = 0; ib32 < 8; ib32++)
        {
            uint indices = BinaryPrimitives.ReadUInt32LittleEndian(p[(2 + 8 * ib32)..]);
            uint signsAndScale = BinaryPrimitives.ReadUInt32LittleEndian(p[(6 + 8 * ib32)..]);
            float db = d * (0.5f + (signsAndScale >> 28)) * 0.25f;
            for (int l = 0; l < 4; l++)
            {
                int signs = IqSigns.Values[(int)((signsAndScale >> (7 * l)) & 127)];
                StoreSigned(Iq2XxsGrid.Values[(int)((indices >> (8 * l)) & 0xFF)], signs, db, y.Slice(32 * ib32 + 8 * l, 8));
            }
        }
    }

    // IQ2_XS: { f16 d; u16 qs[32]; u8 scales[8] }: each u16 a 9-bit grid index and a 7-bit sign index, two scale nibbles per 32 values.
    private static void DecodeIq2Xs(ReadOnlySpan<byte> p, Span<float> y)
    {
        float d = Half16(p);
        for (int ib32 = 0; ib32 < 8; ib32++)
        {
            int scales = p[66 + ib32];
            float db0 = d * (0.5f + (scales & 0xF)) * 0.25f;
            float db1 = d * (0.5f + (scales >> 4)) * 0.25f;
            for (int l = 0; l < 4; l++)
            {
                int q = BinaryPrimitives.ReadUInt16LittleEndian(p[(2 + 2 * (4 * ib32 + l))..]);
                StoreSigned(Iq2XsGrid.Values[q & 511], IqSigns.Values[q >> 9], l < 2 ? db0 : db1, y.Slice(32 * ib32 + 8 * l, 8));
            }
        }
    }

    // IQ2_S: { f16 d; u8 qs[64]; u8 qh[8]; u8 scales[8] }: qs[0..32) low index bits, qs[32..64) sign bytes, qh two high index bits per 8.
    private static void DecodeIq2S(ReadOnlySpan<byte> p, Span<float> y)
    {
        float d = Half16(p);
        for (int ib32 = 0; ib32 < 8; ib32++)
        {
            int scales = p[74 + ib32];
            float db0 = d * (0.5f + (scales & 0xF)) * 0.25f;
            float db1 = d * (0.5f + (scales >> 4)) * 0.25f;
            int high = p[66 + ib32];
            for (int l = 0; l < 4; l++)
            {
                int index = p[2 + 4 * ib32 + l] | ((high << (8 - 2 * l)) & 0x300);
                StoreSigned(Iq2SGrid.Values[index], p[34 + 4 * ib32 + l], l < 2 ? db0 : db1, y.Slice(32 * ib32 + 8 * l, 8));
            }
        }
    }

    // IQ3_XXS: { f16 d; u8 qs[96] }: qs[0..64) 8-bit indices of four-value entries, qs[64..96) per 32 values four 7-bit sign indices and a scale.
    private static void DecodeIq3Xxs(ReadOnlySpan<byte> p, Span<float> y)
    {
        float d = Half16(p);
        var grid = Iq3XxsGrid.Values;
        for (int ib32 = 0; ib32 < 8; ib32++)
        {
            uint word = BinaryPrimitives.ReadUInt32LittleEndian(p[(66 + 4 * ib32)..]);
            float db = d * (0.5f + (word >> 28)) * 0.5f;
            var qs = p.Slice(2 + 8 * ib32, 8);
            for (int l = 0; l < 4; l++)
            {
                int signs = IqSigns.Values[(int)((word >> (7 * l)) & 127)];
                StoreSigned(Join(grid[qs[2 * l]], grid[qs[2 * l + 1]]), signs, db, y.Slice(32 * ib32 + 8 * l, 8));
            }
        }
    }

    // IQ3_S: { f16 d; u8 qs[64]; u8 qh[8]; u8 signs[32]; u8 scales[4] }: nine-bit indices of four-value entries, odd scales 1 + 2s.
    private static void DecodeIq3S(ReadOnlySpan<byte> p, Span<float> y)
    {
        float d = Half16(p);
        var grid = Iq3SGrid.Values;
        for (int ib32 = 0; ib32 < 8; ib32++)
        {
            int scale = p[106 + ib32 / 2] >> (4 * (ib32 % 2)) & 0xF;
            float db = d * (1 + 2 * scale);
            int high = p[66 + ib32];
            var qs = p.Slice(2 + 8 * ib32, 8);
            for (int l = 0; l < 4; l++)
            {
                int lo = qs[2 * l] | ((high << (8 - 2 * l)) & 256);
                int hi = qs[2 * l + 1] | ((high << (7 - 2 * l)) & 256);
                StoreSigned(Join(grid[lo], grid[hi]), p[74 + 4 * ib32 + l], db, y.Slice(32 * ib32 + 8 * l, 8));
            }
        }
    }

    // Eight values: (signed byte j of an IQ1 grid entry + delta) times scale.
    private static void StoreIq1(int index, float delta, float scale, Span<float> y)
    {
        ulong entry = Iq1SGrid.Values[index];
        for (int j = 0; j < 8; j++)
        {
            y[j] = ((sbyte)(entry >> (8 * j)) + delta) * scale;
        }
    }

    // IQ1_S: { f16 d; u8 qs[32]; u16 qh[8] }: per 32 values three high index bits per group of eight, a scale 2s + 1 and a delta sign.
    private static void DecodeIq1S(ReadOnlySpan<byte> p, Span<float> y)
    {
        float d = Half16(p);
        for (int ib = 0; ib < 8; ib++)
        {
            int qh = BinaryPrimitives.ReadUInt16LittleEndian(p[(34 + 2 * ib)..]);
            float dl = d * (2 * ((qh >> 12) & 7) + 1);
            float delta = (qh & 0x8000) != 0 ? -Iq1Delta : Iq1Delta;
            for (int l = 0; l < 4; l++)
            {
                StoreIq1(p[2 + 4 * ib + l] | (((qh >> (3 * l)) & 7) << 8), delta, dl, y.Slice(32 * ib + 8 * l, 8));
            }
        }
    }

    // IQ1_M: { u8 qs[32]; u8 qh[16]; u16 scales[4] }: a 3-bit scale per 16 values and a delta sign per 8; the f16 super-scale is spread
    // over the top nibbles of the four scale words.
    private static void DecodeIq1M(ReadOnlySpan<byte> p, Span<float> y)
    {
        Span<int> sc = stackalloc int[4];
        for (int i = 0; i < 4; i++)
        {
            sc[i] = BinaryPrimitives.ReadUInt16LittleEndian(p[(48 + 2 * i)..]);
        }

        int scaleBits = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
        float d = (float)BitConverter.UInt16BitsToHalf((ushort)scaleBits);
        for (int ib = 0; ib < 8; ib++)
        {
            float dl1 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2))) & 7) + 1);
            float dl2 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 7) + 1);
            var qs = p.Slice(4 * ib, 4);
            int qh0 = p[32 + 2 * ib];
            int qh1 = p[33 + 2 * ib];
            var o = y.Slice(32 * ib, 32);
            StoreIq1(qs[0] | ((qh0 << 8) & 0x700), (qh0 & 0x08) != 0 ? -Iq1Delta : Iq1Delta, dl1, o[..8]);
            StoreIq1(qs[1] | ((qh0 << 4) & 0x700), (qh0 & 0x80) != 0 ? -Iq1Delta : Iq1Delta, dl1, o.Slice(8, 8));
            StoreIq1(qs[2] | ((qh1 << 8) & 0x700), (qh1 & 0x08) != 0 ? -Iq1Delta : Iq1Delta, dl2, o.Slice(16, 8));
            StoreIq1(qs[3] | ((qh1 << 4) & 0x700), (qh1 & 0x80) != 0 ? -Iq1Delta : Iq1Delta, dl2, o.Slice(24, 8));
        }
    }

    // The base-3 digit n of a byte holding five ternary values, as -1, 0 or 1.
    private static int Trit(byte b, int n)
    {
        int power = n switch { 0 => 1, 1 => 3, 2 => 9, 3 => 27, _ => 81 };
        return ((((b * power) & 0xFF) * 3) >> 8) - 1;
    }

    // TQ1_0: { u8 qs[48]; u8 qh[4]; f16 d }: five base-3 digits per byte (qs[0..32) values n·32 + m, qs[32..48) 160 + n·16 + m, qh 240 + n·4 + j).
    private static void DecodeTq1(ReadOnlySpan<byte> p, Span<float> y)
    {
        float d = Half16(p[52..]);
        for (int n = 0; n < 5; n++)
        {
            for (int m = 0; m < 32; m++)
            {
                y[n * 32 + m] = Trit(p[m], n) * d;
            }

            for (int m = 0; m < 16; m++)
            {
                y[160 + n * 16 + m] = Trit(p[32 + m], n) * d;
            }
        }

        for (int n = 0; n < 4; n++)
        {
            for (int j = 0; j < 4; j++)
            {
                y[240 + n * 4 + j] = Trit(p[48 + j], n) * d;
            }
        }
    }

    // TQ2_0: { u8 qs[64]; f16 d }: value (j·4 + l)·32 + m is ((qs[j·32 + m] >> 2l) & 3) - 1, times d.
    private static void DecodeTq2(ReadOnlySpan<byte> p, Span<float> y)
    {
        float d = Half16(p[64..]);
        for (int j = 0; j < 2; j++)
        {
            for (int l = 0; l < 4; l++)
            {
                for (int m = 0; m < 32; m++)
                {
                    y[128 * j + 32 * l + m] = (((p[32 * j + m] >> (2 * l)) & 3) - 1) * d;
                }
            }
        }
    }

    // E2M1 codes (doubled magnitudes), low nibbles first then high nibbles, times the halved scale.
    private static void DecodeFp4(float d, ReadOnlySpan<byte> qs, Span<float> y)
    {
        for (int j = 0; j < qs.Length; j++)
        {
            y[j] = s_fp4Doubled[qs[j] & 0xF] * d;
            y[j + qs.Length] = s_fp4Doubled[qs[j] >> 4] * d;
        }
    }

    // The MXFP4 E8M0 scale halved (the stored E2M1 values are doubled): 2^(e - 128), the two smallest exponents as denormals.
    private static float E8M0Half(byte e)
        => BitConverter.UInt32BitsToSingle(e < 2 ? 0x00200000u << e : (uint)(e - 1) << 23);

    // The NVFP4 UE4M3 scale halved, 0 and 0x7F (NaN) as zero.
    private static float Ue4M3Half(byte x)
    {
        if (x == 0 || x == 0x7F)
        {
            return 0f;
        }

        int exp = (x >> 3) & 0xF;
        int man = x & 7;
        float raw = exp == 0 ? MathF.ScaleB(man, -9) : MathF.ScaleB(1f + man / 8f, exp - 7);
        return raw * 0.5f;
    }
}
