using Sharplight.Tensors.Grids;

namespace Sharplight.Cpu;

// 4-bit float lookup blocks (FP4_B64), f32 absmax int8 (Q8A_G32) and the f32-scaled ternary 2-bit code (Q2S_G128).
internal static unsafe partial class Quant
{
    /// <summary>The 4-bit float code book of <see cref="Sharplight.Tensors.DataType.FP4_B64"/>, in code order.</summary>
    internal static readonly float[] Fp4Values = Fp4CodeBook.Values.ToArray();

    /// <summary>FP4_B64: 64 values = { f32 d; u8 qs[32] } laid out like NF4_B64; value = d * FP4[q].</summary>
    private static void FP4_B64(byte* p, long nb, float* dst) => LookupB64(p, nb, Fp4Values, dst);

    /// <summary>Q8A_G32: 32 values = { f32 d; i8 q[32] }; value = q * d * (1/127), in that order.</summary>
    private static void Q8A_G32(byte* p, long nb, float* dst)
    {
        for (long b = 0; b < nb; b++, p += 36, dst += 32)
        {
            float d = *(float*)p;
            sbyte* q = (sbyte*)(p + 4);
            for (int j = 0; j < 32; j++)
            {
                dst[j] = q[j] * d * 7.874015718698502e-3f;
            }
        }
    }

    /// <summary>Q2S_G128: 128 values = { f32 d; u8 qs[32] }, byte j holding values j, 32 + j, 64 + j, 96 + j in bits 6, 4, 2, 0; value = (code - 1) * d.</summary>
    private static void Q2S_G128(byte* p, long nb, float* dst)
    {
        for (long b = 0; b < nb; b++, p += 36, dst += 128)
        {
            float d = *(float*)p;
            byte* q = p + 4;
            for (int j = 0; j < 32; j++)
            {
                dst[j] = (((q[j] >> 6) & 3) - 1) * d;
                dst[32 + j] = (((q[j] >> 4) & 3) - 1) * d;
                dst[64 + j] = (((q[j] >> 2) & 3) - 1) * d;
                dst[96 + j] = ((q[j] & 3) - 1) * d;
            }
        }
    }
}
