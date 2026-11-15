using System.Runtime.Intrinsics;
using Sharplight.Tensors;
using Sharplight.Tensors.Grids;

namespace Sharplight.Cpu;

// Integer codes with a group scale and zero point (GPTQ / AWQ style), the NF4 lookup format (bitsandbytes) and NVFP4 with a global scale.
internal static unsafe partial class Quant
{
    /// <summary>The bitsandbytes NF4 code book (normal-distribution quantiles normalized to [-1, 1]), in code order.</summary>
    internal static readonly float[] Nf4Values = Nf4CodeBook.Values.ToArray();

    /// <summary>The grouped integer, lookup or global-scale FP4 format of <paramref name="t"/>, or null when it is none of them.</summary>
    private static WeightFormat? GroupedFormatOf(DataType t)
    {
        if (t == DataType.NVFP4_GS)
        {
            return WeightFormat.NVFP4_GS;
        }

        if (t == DataType.Q4Z_G32)
        {
            return WeightFormat.Q4Z_G32;
        }

        if (t == DataType.Q4Z_G128)
        {
            return WeightFormat.Q4Z_G128;
        }

        if (t == DataType.Q8Z_G32)
        {
            return WeightFormat.Q8Z_G32;
        }

        if (t == DataType.Q8Z_G128)
        {
            return WeightFormat.Q8Z_G128;
        }

        if (t == DataType.NF4_B64)
        {
            return WeightFormat.NF4_B64;
        }

        return null;
    }

    /// <summary>
    /// Q4Z: <paramref name="group"/> values = { f16 d; f16 z; u8 qs[group / 2] }; each 32-value run of 16 bytes holds value j in the
    /// low nibble of byte j and value j + 16 in the high one; value = (q - z) * d.
    /// </summary>
    private static void Q4Z(byte* p, long nb, int group, float* dst)
    {
        var mask = Vector128.Create((byte)0x0F);
        int blockBytes = 4 + group / 2;
        for (long b = 0; b < nb; b++, p += blockBytes, dst += group)
        {
            float d = H(p);
            float offset = -H(p + 2) * d;
            for (int run = 0; run < group / 32; run++)
            {
                var q = Vector128.Load(p + 4 + 16 * run);
                Simd.StoreU8x16(q & mask, d, offset, dst + 32 * run);
                Simd.StoreU8x16(Vector128.ShiftRightLogical(q, 4), d, offset, dst + 32 * run + 16);
            }
        }
    }

    /// <summary>Q8Z: <paramref name="group"/> values = { f16 d; f16 z; u8 q[group] }; value = (q - z) * d.</summary>
    private static void Q8Z(byte* p, long nb, int group, float* dst)
    {
        int blockBytes = 4 + group;
        for (long b = 0; b < nb; b++, p += blockBytes, dst += group)
        {
            float d = H(p);
            float offset = -H(p + 2) * d;
            for (int j = 0; j < group; j += 16)
            {
                Simd.StoreU8x16(Vector128.Load(p + 4 + j), d, offset, dst + j);
            }
        }
    }

    /// <summary>NF4_B64: 64 values = { f32 d; u8 qs[32] }, two 32-value runs laid out like Q4_0's codes; value = d * NF4[q].</summary>
    private static void NF4_B64(byte* p, long nb, float* dst)
    {
        fixed (float* table = Nf4Values)
        {
            for (long b = 0; b < nb; b++, p += 36, dst += 64)
            {
                float d = *(float*)p;
                for (int run = 0; run < 2; run++)
                {
                    byte* q = p + 4 + 16 * run;
                    float* y = dst + 32 * run;
                    for (int j = 0; j < 16; j++)
                    {
                        y[j] = table[q[j] & 0xF] * d;
                        y[j + 16] = table[q[j] >> 4] * d;
                    }
                }
            }
        }
    }
}
