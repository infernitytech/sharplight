using System.Runtime.Intrinsics;

namespace Sharplight.Cpu;

// ggml's ternary (BitNet) formats TQ1_0 and TQ2_0: values are -d, 0 or +d.
internal static unsafe partial class Quant
{
    /// <summary>The base-3 digit n of byte b, ((b * 3^n mod 256) * 3) >> 8, minus one, for n in 0..4: [n * 256 + b].</summary>
    private static readonly sbyte[] s_trits = BuildTrits();

    private static sbyte[] BuildTrits()
    {
        var table = new sbyte[5 * 256];
        int pow3 = 1;
        for (int n = 0; n < 5; n++, pow3 *= 3)
        {
            for (int b = 0; b < 256; b++)
            {
                table[n * 256 + b] = (sbyte)((((b * pow3) & 0xFF) * 3 >> 8) - 1);
            }
        }

        return table;
    }

    /// <summary>
    /// TQ1_0: 256 values = { u8 qs[48]; u8 qh[4]; f16 d } (54 bytes): five base-3 digits per byte. qs[0..32) gives values n*32+m (digit n of
    /// byte m), qs[32..48) gives 160+n*16+m, and the four qh bytes give 240+n*4+j (digits 0..3).
    /// </summary>
    private static void TQ1_0(byte* p, long nb, float* dst)
    {
        fixed (sbyte* trits = s_trits)
        {
            for (long b = 0; b < nb; b++, p += 54, dst += 256)
            {
                float d = H(p + 52);
                for (int n = 0; n < 5; n++)
                {
                    sbyte* t = trits + n * 256;
                    for (int m = 0; m < 32; m++)
                    {
                        dst[n * 32 + m] = t[p[m]] * d;
                    }

                    for (int m = 0; m < 16; m++)
                    {
                        dst[160 + n * 16 + m] = t[p[32 + m]] * d;
                    }
                }

                for (int n = 0; n < 4; n++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        dst[240 + n * 4 + j] = trits[n * 256 + p[48 + j]] * d;
                    }
                }
            }
        }
    }

    /// <summary>TQ2_0: 256 values = { u8 qs[64]; f16 d } (66 bytes); value (j*4 + l)*32 + m = ((qs[j*32 + m] >> 2l) &amp; 3) - 1, times d, for j in 0..1.</summary>
    private static void TQ2_0(byte* p, long nb, float* dst)
    {
        var three = Vector128.Create((byte)3);
        var one = Vector128.Create((byte)1);
        for (long b = 0; b < nb; b++, p += 66, dst += 256)
        {
            float d = H(p + 64);
            for (int j = 0; j < 2; j++)
            {
                for (int l = 0; l < 4; l++)
                {
                    for (int h = 0; h < 2; h++)
                    {
                        var v = (Vector128.ShiftRightLogical(Vector128.Load(p + 32 * j + 16 * h), 2 * l) & three) - one;
                        StoreI8Scaled(v.AsSByte(), d, dst + 128 * j + 32 * l + 16 * h);
                    }
                }
            }
        }
    }
}
