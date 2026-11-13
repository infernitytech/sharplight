using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace Sharplight.Cpu;

// 4-bit floating point formats: MXFP4 (OCP microscaling, shared E8M0 scale per 32) and NVFP4 (UE4M3 scale per 16).
internal static unsafe partial class Quant
{
    /// <summary>The E2M1 magnitudes doubled (ggml kvalues_fp4): code 0..7 positive, 8..15 negative.</summary>
    private static readonly sbyte[] s_fp4Values = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

    private static readonly float[] s_ue4m3 = BuildUe4m3();

    /// <summary>UE4M3 scale byte (4 exponent bits, bias 7, 3 mantissa bits) decoded as ggml does: halved, and 0 / 0x7F (NaN) read as zero.</summary>
    private static float[] BuildUe4m3()
    {
        var table = new float[256];
        for (int x = 0; x < 256; x++)
        {
            if (x == 0 || x == 0x7F)
            {
                continue;
            }

            int exp = (x >> 3) & 0xF;
            int man = x & 7;
            float raw = exp == 0 ? MathF.ScaleB(man, -9) : MathF.ScaleB(1f + man / 8f, exp - 7);
            table[x] = raw * 0.5f;
        }

        return table;
    }

    /// <summary>The MXFP4 scale: 2^(e-127) / 2 (the stored E2M1 values are doubled), with the two smallest exponents as denormals.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float E8M0Half(byte e)
    {
        uint bits = e < 2 ? 0x00200000u << e : (uint)(e - 1) << 23;
        return BitConverter.UInt32BitsToSingle(bits);
    }

    /// <summary>MXFP4: 32 values = { u8 e; u8 qs[16] } (17 bytes); value j&lt;16: fp4(low nibble of qs[j]), j&gt;=16: fp4(high nibble of qs[j-16]), times the E8M0 scale.</summary>
    private static void MXFP4(byte* p, long nb, float* dst)
    {
        var table = Vector128.Create(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12).AsSByte();
        var mask = Vector128.Create((byte)0x0F);
        for (long b = 0; b < nb; b++, p += 17, dst += 32)
        {
            float d = E8M0Half(p[0]);
            var q = Vector128.Load(p + 1);
            StoreI8Scaled(Vector128.Shuffle(table, (q & mask).AsSByte()), d, dst);
            StoreI8Scaled(Vector128.Shuffle(table, Vector128.ShiftRightLogical(q, 4).AsSByte()), d, dst + 16);
        }
    }

    /// <summary>NVFP4: 64 values = { u8 d[4]; u8 qs[32] } (36 bytes); four 16-value sub-blocks with a UE4M3 scale each, laid out like MXFP4 halves (low nibbles first).</summary>
    private static void NVFP4(byte* p, long nb, float* dst)
    {
        fixed (float* scaleTable = s_ue4m3)
        fixed (sbyte* fp4 = s_fp4Values)
        {
            for (long b = 0; b < nb; b++, p += 36, dst += 64)
            {
                for (int s = 0; s < 4; s++)
                {
                    float d = scaleTable[p[s]];
                    byte* q = p + 4 + 8 * s;
                    float* y = dst + 16 * s;
                    for (int j = 0; j < 8; j++)
                    {
                        y[j] = fp4[q[j] & 0xF] * d;
                        y[j + 8] = fp4[q[j] >> 4] * d;
                    }
                }
            }
        }
    }
}
