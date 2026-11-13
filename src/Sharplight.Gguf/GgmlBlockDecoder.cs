using System.Buffers.Binary;
using Sharplight.Tensors;

namespace Sharplight.Gguf;

/// <summary>
/// Managed reference decoder of the common ggml storage formats (float types, the 32-element legacy blocks and the
/// Q4_K / Q5_K / Q6_K super-blocks) to f32. Used at import time for the few tensors that need a layout change a byte-level
/// row reorder cannot express (e.g. transposing a quantized matrix); compute paths use the providers' own kernels.
/// </summary>
internal static class GgmlBlockDecoder
{
    /// <summary>Decodes a whole tensor of <paramref name="type"/> to f32.</summary>
    /// <param name="type">Storage type of <paramref name="data"/>.</param>
    /// <param name="data">The tensor bytes (a whole number of blocks).</param>
    /// <returns>The decoded elements in storage order.</returns>
    /// <exception cref="NotSupportedException">The type has no managed decoder.</exception>
    public static float[] Decode(DataType type, ReadOnlySpan<byte> data)
    {
        long blocks = data.Length / type.BlockBytes;
        var dst = new float[blocks * type.BlockSize];
        for (long b = 0; b < blocks; b++)
        {
            var block = data.Slice((int)(b * type.BlockBytes), type.BlockBytes);
            DecodeBlock(type, block, dst.AsSpan((int)(b * type.BlockSize), type.BlockSize));
        }

        return dst;
    }

    private static void DecodeBlock(DataType type, ReadOnlySpan<byte> block, Span<float> y)
    {
        if (type == DataType.F32)
        {
            y[0] = BinaryPrimitives.ReadSingleLittleEndian(block);
        }
        else if (type == DataType.F16)
        {
            y[0] = Half16(block);
        }
        else if (type == DataType.BF16)
        {
            y[0] = BitConverter.UInt32BitsToSingle((uint)BinaryPrimitives.ReadUInt16LittleEndian(block) << 16);
        }
        else if (type == DataType.Q8_0)
        {
            float d = Half16(block);
            for (int j = 0; j < 32; j++)
            {
                y[j] = (sbyte)block[2 + j] * d;
            }
        }
        else if (type == DataType.Q4_0 || type == DataType.Q4_1)
        {
            DecodeQ4(type == DataType.Q4_1, block, y);
        }
        else if (type == DataType.Q5_0 || type == DataType.Q5_1)
        {
            DecodeQ5(type == DataType.Q5_1, block, y);
        }
        else if (type == DataType.Q4_K)
        {
            DecodeQ4K(block, y);
        }
        else if (type == DataType.Q5_K)
        {
            DecodeQ5K(block, y);
        }
        else if (type == DataType.Q6_K)
        {
            DecodeQ6K(block, y);
        }
        else
        {
            throw new NotSupportedException($"No managed decoder for {type}.");
        }
    }

    private static float Half16(ReadOnlySpan<byte> bytes) => (float)BinaryPrimitives.ReadHalfLittleEndian(bytes);

    // Q4_0: d, 16 bytes of nibbles (low nibbles = elements 0..15, high = 16..31), value (q - 8)·d. Q4_1: d, m, q·d + m.
    private static void DecodeQ4(bool hasMin, ReadOnlySpan<byte> block, Span<float> y)
    {
        float d = Half16(block);
        float m = hasMin ? Half16(block[2..]) : 0f;
        int offset = hasMin ? 0 : 8;
        var qs = block[(hasMin ? 4 : 2)..];
        for (int j = 0; j < 16; j++)
        {
            y[j] = ((qs[j] & 0xF) - offset) * d + m;
            y[j + 16] = ((qs[j] >> 4) - offset) * d + m;
        }
    }

    // Q5_0 / Q5_1: as Q4 with a fifth bit per element from a 32-bit mask; offset 16 without a minimum.
    private static void DecodeQ5(bool hasMin, ReadOnlySpan<byte> block, Span<float> y)
    {
        float d = Half16(block);
        float m = hasMin ? Half16(block[2..]) : 0f;
        int header = hasMin ? 4 : 2;
        uint qh = BinaryPrimitives.ReadUInt32LittleEndian(block[header..]);
        var qs = block[(header + 4)..];
        int offset = hasMin ? 0 : 16;
        for (int j = 0; j < 16; j++)
        {
            int high0 = (int)((qh >> j) << 4) & 0x10;
            int high1 = (int)(qh >> (j + 12)) & 0x10;
            y[j] = (((qs[j] & 0xF) | high0) - offset) * d + m;
            y[j + 16] = (((qs[j] >> 4) | high1) - offset) * d + m;
        }
    }

    // 6-bit scale and minimum j of the 12 packed scale bytes of a Q4_K / Q5_K super-block.
    private static (int Scale, int Min) ScaleMin(int j, ReadOnlySpan<byte> q)
    {
        if (j < 4)
        {
            return (q[j] & 63, q[j + 4] & 63);
        }

        return ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }

    private static void DecodeQ4K(ReadOnlySpan<byte> block, Span<float> y)
    {
        float d = Half16(block);
        float dmin = Half16(block[2..]);
        var scales = block.Slice(4, 12);
        var q = block[16..];
        int o = 0;
        for (int j = 0, sub = 0; j < 256; j += 64, sub += 2)
        {
            var (sc1, m1) = ScaleMin(sub, scales);
            var (sc2, m2) = ScaleMin(sub + 1, scales);
            for (int l = 0; l < 32; l++)
            {
                y[o + l] = d * sc1 * (q[l] & 0xF) - dmin * m1;
                y[o + 32 + l] = d * sc2 * (q[l] >> 4) - dmin * m2;
            }

            q = q[32..];
            o += 64;
        }
    }

    private static void DecodeQ5K(ReadOnlySpan<byte> block, Span<float> y)
    {
        float d = Half16(block);
        float dmin = Half16(block[2..]);
        var scales = block.Slice(4, 12);
        var qh = block.Slice(16, 32);
        var ql = block[48..];
        int o = 0;
        int u1 = 1;
        int u2 = 2;
        for (int j = 0, sub = 0; j < 256; j += 64, sub += 2)
        {
            var (sc1, m1) = ScaleMin(sub, scales);
            var (sc2, m2) = ScaleMin(sub + 1, scales);
            for (int l = 0; l < 32; l++)
            {
                y[o + l] = d * sc1 * ((ql[l] & 0xF) + ((qh[l] & u1) != 0 ? 16 : 0)) - dmin * m1;
                y[o + 32 + l] = d * sc2 * ((ql[l] >> 4) + ((qh[l] & u2) != 0 ? 16 : 0)) - dmin * m2;
            }

            ql = ql[32..];
            o += 64;
            u1 <<= 2;
            u2 <<= 2;
        }
    }

    private static void DecodeQ6K(ReadOnlySpan<byte> block, Span<float> y)
    {
        var ql = block[..128];
        var qh = block.Slice(128, 64);
        var sc = block.Slice(192, 16);
        float d = Half16(block[208..]);
        for (int n = 0; n < 2; n++)
        {
            int o = n * 128;
            var l0 = ql[(n * 64)..];
            var h0 = qh[(n * 32)..];
            int s0 = n * 8;
            for (int l = 0; l < 32; l++)
            {
                int scaleIndex = s0 + l / 16;
                int q1 = ((l0[l] & 0xF) | ((h0[l] & 3) << 4)) - 32;
                int q2 = ((l0[l + 32] & 0xF) | (((h0[l] >> 2) & 3) << 4)) - 32;
                int q3 = ((l0[l] >> 4) | (((h0[l] >> 4) & 3) << 4)) - 32;
                int q4 = ((l0[l + 32] >> 4) | (((h0[l] >> 6) & 3) << 4)) - 32;
                y[o + l] = d * (sbyte)sc[scaleIndex] * q1;
                y[o + l + 32] = d * (sbyte)sc[scaleIndex + 2] * q2;
                y[o + l + 64] = d * (sbyte)sc[scaleIndex + 4] * q3;
                y[o + l + 96] = d * (sbyte)sc[scaleIndex + 6] * q4;
            }
        }
    }
}
