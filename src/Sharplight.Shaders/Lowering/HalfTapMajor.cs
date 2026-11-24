using System.Runtime.InteropServices;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

/// <summary>Host conversion of convolution weights to the channels-last 16-bit form conv_cl reads (see <see cref="GraphLowerer"/>).</summary>
internal static class HalfTapMajor
{
    /// <summary>
    /// Rows of [<paramref name="channels"/>][<paramref name="taps"/>] values of <paramref name="from"/> (f32, f16 or bf16) as rows of
    /// [taps][channels] values of <paramref name="to"/> (f16 or bf16, rounded to nearest even).
    /// </summary>
    public static byte[] Convert(byte[] source, DataType from, DataType to, long rows, long channels, long taps)
    {
        long k = channels * taps;
        var values = from == DataType.F32 ? MemoryMarshal.Cast<byte, float>(source.AsSpan()).ToArray() : Widen(source, from == DataType.F16);
        var result = new byte[checked(rows * k * 2)];
        var dst = MemoryMarshal.Cast<byte, ushort>(result.AsSpan());
        bool bf16 = to == DataType.BF16;
        for (long r = 0; r < rows; r++)
        {
            for (long c = 0; c < channels; c++)
            {
                for (long t = 0; t < taps; t++)
                {
                    float v = values[r * k + c * taps + t];
                    dst[(int)(r * k + t * channels + c)] = bf16 ? BFloat16Bits(v) : BitConverter.HalfToUInt16Bits((Half)v);
                }
            }
        }

        return result;
    }

    /// <summary>f16 (<paramref name="half"/>) or bf16 values as floats.</summary>
    private static float[] Widen(byte[] source, bool half)
    {
        var bits = MemoryMarshal.Cast<byte, ushort>(source.AsSpan());
        var values = new float[bits.Length];
        for (int i = 0; i < bits.Length; i++)
        {
            values[i] = half ? (float)BitConverter.UInt16BitsToHalf(bits[i]) : BitConverter.Int32BitsToSingle(bits[i] << 16);
        }

        return values;
    }

    /// <summary>bf16 bits of a finite <paramref name="v"/>, rounded to nearest even.</summary>
    private static ushort BFloat16Bits(float v)
    {
        uint u = BitConverter.SingleToUInt32Bits(v);
        return (ushort)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
    }
}
