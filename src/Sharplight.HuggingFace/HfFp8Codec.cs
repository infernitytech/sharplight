using System.Buffers.Binary;
using Sharplight.Tensors;

namespace Sharplight.HuggingFace;

/// <summary>Host-side decoding of the FP8 storage types (<see cref="DataType.F8E4M3"/>, <see cref="DataType.F8E5M2"/> and the block-scaled E4M3 formats).</summary>
internal static class HfFp8Codec
{
    private static readonly float[] s_e4m3 = [.. Enumerable.Range(0, 256).Select(code => E4m3ToFloat((byte)code))];

    /// <summary>Whether <paramref name="type"/> is one of the FP8 storage types.</summary>
    public static bool IsFp8(DataType type)
        => type == DataType.F8E4M3 || type == DataType.F8E5M2 || type == DataType.F8E4M3_B32 || type == DataType.F8E4M3_B128;

    /// <summary>The value of an E4M3FN code: bias 7, exponent 0 subnormal (m * 2^-9), no infinity, 0x7F / 0xFF NaN.</summary>
    public static float E4m3ToFloat(byte code)
    {
        int exponent = (code >> 3) & 0xF;
        int mantissa = code & 7;
        if (exponent == 0xF && mantissa == 7)
        {
            return float.NaN;
        }

        float magnitude = exponent == 0 ? MathF.ScaleB(mantissa, -9) : MathF.ScaleB(1f + mantissa / 8f, exponent - 7);
        return (code & 0x80) != 0 ? -magnitude : magnitude;
    }

    /// <summary>The value of an E4M3FN code from a lookup table.</summary>
    public static float E4m3(byte code) => s_e4m3[code];

    /// <summary>Decodes the bytes of an FP8 tensor of <paramref name="type"/> to F32 values.</summary>
    /// <exception cref="NotSupportedException"><paramref name="type"/> is not an FP8 type.</exception>
    public static float[] Decode(DataType type, ReadOnlySpan<byte> bytes)
    {
        if (type == DataType.F8E4M3)
        {
            var values = new float[bytes.Length];
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = s_e4m3[bytes[i]];
            }

            return values;
        }

        if (type == DataType.F8E5M2)
        {
            var values = new float[bytes.Length];
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = (float)BitConverter.UInt16BitsToHalf((ushort)(bytes[i] << 8));
            }

            return values;
        }

        if (type == DataType.F8E4M3_B32 || type == DataType.F8E4M3_B128)
        {
            int block = type.BlockSize;
            int blocks = bytes.Length / type.BlockBytes;
            var values = new float[(long)blocks * block];
            for (int b = 0; b < blocks; b++)
            {
                var blockBytes = bytes.Slice(b * type.BlockBytes, type.BlockBytes);
                float scale = BinaryPrimitives.ReadSingleLittleEndian(blockBytes);
                for (int i = 0; i < block; i++)
                {
                    values[b * block + i] = s_e4m3[blockBytes[4 + i]] * scale;
                }
            }

            return values;
        }

        throw new NotSupportedException($"{type} is not an FP8 type.");
    }

    /// <summary>The bfloat16 bits of <paramref name="value"/>, rounded to nearest even (NaN stays NaN).</summary>
    public static ushort ToBFloat16(float value)
    {
        uint bits = BitConverter.SingleToUInt32Bits(value);
        if (float.IsNaN(value))
        {
            return (ushort)((bits >> 16) | 0x40);
        }

        return (ushort)((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
    }
}
