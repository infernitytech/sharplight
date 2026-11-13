using System.Runtime.CompilerServices;
using Sharplight.Tensors;

namespace Sharplight.Cpu;

// 8-bit floating point weights (OCP FP8): plain E4M3FN / E5M2 values and E4M3 codes in blocks of 32 or 128 with an f32 scale.
internal static unsafe partial class Quant
{
    private static readonly float[] s_e4m3 = BuildE4m3();
    private static readonly float[] s_e5m2 = BuildE5m2();

    /// <summary>The FP8 weight format of <paramref name="t"/>, or null when it is not an FP8 type.</summary>
    private static WeightFormat? Fp8FormatOf(DataType t)
    {
        if (t == DataType.F8E4M3)
        {
            return WeightFormat.F8E4M3;
        }

        if (t == DataType.F8E5M2)
        {
            return WeightFormat.F8E5M2;
        }

        if (t == DataType.F8E4M3_B32)
        {
            return WeightFormat.F8E4M3_B32;
        }

        if (t == DataType.F8E4M3_B128)
        {
            return WeightFormat.F8E4M3_B128;
        }

        return null;
    }

    /// <summary>Decodes one E4M3FN byte: bias 7, exponent 0 subnormal (m * 2^-9), no infinity, 0x7F / 0xFF NaN.</summary>
    internal static float E4m3ToFloat(byte x)
    {
        int exp = (x >> 3) & 0xF;
        int man = x & 7;
        if (exp == 0xF && man == 7)
        {
            return float.NaN;
        }

        float magnitude = exp == 0 ? MathF.ScaleB(man, -9) : MathF.ScaleB(1f + man / 8f, exp - 7);
        return (x & 0x80) != 0 ? -magnitude : magnitude;
    }

    private static float[] BuildE4m3()
    {
        var table = new float[256];
        for (int x = 0; x < 256; x++)
        {
            table[x] = E4m3ToFloat((byte)x);
        }

        return table;
    }

    /// <summary>E5M2 is the upper byte of an IEEE half.</summary>
    private static float[] BuildE5m2()
    {
        var table = new float[256];
        for (int x = 0; x < 256; x++)
        {
            table[x] = (float)BitConverter.UInt16BitsToHalf((ushort)(x << 8));
        }

        return table;
    }

    /// <summary>Dequantizes <paramref name="count"/> values of an FP8 format (a multiple of its block size).</summary>
    private static void DequantizeFp8(WeightFormat f, byte* src, long count, float* dst)
    {
        switch (f)
        {
            case WeightFormat.F8E4M3:
                Lookup(s_e4m3, src, count, 1f, dst);
                break;
            case WeightFormat.F8E5M2:
                Lookup(s_e5m2, src, count, 1f, dst);
                break;
            case WeightFormat.F8E4M3_B32:
                ScaledE4m3Blocks(src, count, 32, dst);
                break;
            case WeightFormat.F8E4M3_B128:
                ScaledE4m3Blocks(src, count, 128, dst);
                break;
            default:
                throw new NotSupportedException(f.ToString());
        }
    }

    /// <summary>Blocks of { f32 d; u8 q[block] }: value i = d * e4m3(q[i]).</summary>
    private static void ScaledE4m3Blocks(byte* p, long count, int block, float* dst)
    {
        long blocks = count / block;
        for (long b = 0; b < blocks; b++, p += 4 + block, dst += block)
        {
            Lookup(s_e4m3, p + 4, block, Unsafe.ReadUnaligned<float>(p), dst);
        }
    }

    private static void Lookup(float[] table, byte* codes, long count, float scale, float* dst)
    {
        fixed (float* values = table)
        {
            for (long i = 0; i < count; i++)
            {
                dst[i] = values[codes[i]] * scale;
            }
        }
    }
}
