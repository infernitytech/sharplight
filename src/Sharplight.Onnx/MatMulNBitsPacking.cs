using Sharplight.Tensors;

namespace Sharplight.Onnx;

/// <summary>
/// Repacks <c>com.microsoft:MatMulNBits</c> weights ([N, blocks, block_size · bits / 8] codes, one scale per block, optional
/// packed uint8 zero points) into the engine's 32-value block formats, so the backends read 4.5 / 5 / 8.5 bits per weight
/// instead of a dequantized f32 copy:
/// <list type="bullet">
/// <item>4 bits, no zero points (implicit 8): <see cref="DataType.Q4_0"/>, value = (q - 8) · d.</item>
/// <item>4 bits with zero points: <see cref="DataType.Q4_1"/>, value = q · d + m with m = -z · d.</item>
/// <item>8 bits, no zero points (implicit 128): <see cref="DataType.Q8_0"/>, value = (q - 128) · d.</item>
/// </list>
/// Blocks larger than 32 (any multiple of 32) are split into 32-value blocks with the same scale. Scales are stored as f16:
/// exact for f16 models; f32 scales that f16 cannot hold exactly (and the Q4_1 minimum -z · d) are only rounded when the import
/// allows it (<see cref="OnnxImportOptions.AllowHalfBlockScales"/>), otherwise the weight is dequantized to f32.
/// </summary>
internal static class MatMulNBitsPacking
{
    private const int BlockValues = 32;

    /// <summary>Largest relative error of an f32 scale stored as f16 before the packing is refused (subnormal or out-of-range scales).</summary>
    private const double MaxScaleRounding = 1.0 / 1024;

    /// <summary>
    /// The block format the weight packs into, or null when it has to be dequantized to f32. Zero points need
    /// <paramref name="allowRounding"/>: the Q4_1 minimum -z · d is rounded to f16.
    /// </summary>
    public static DataType? TargetType(long k, long bits, long blockSize, bool hasZeroPoints, bool allowRounding)
    {
        if (k % BlockValues != 0 || blockSize % BlockValues != 0)
        {
            return null;
        }

        return (bits, hasZeroPoints) switch
        {
            (4, false) => DataType.Q4_0,
            (4, true) when allowRounding => DataType.Q4_1,
            (8, false) => DataType.Q8_0,
            _ => null,
        };
    }

    /// <summary>
    /// True when every scale converts to f16 exactly, or, with <paramref name="allowRounding"/>, within <see cref="MaxScaleRounding"/>
    /// (refusing subnormal or out-of-range scales either way).
    /// </summary>
    public static bool ScalesFitHalf(double[] scales, bool allowRounding)
    {
        foreach (double s in scales)
        {
            double back = (double)(Half)s;
            double bound = allowRounding ? MaxScaleRounding * Math.Abs(s) : 0;
            if (!double.IsFinite(back) || Math.Abs(back - s) > bound)
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Packs the codes into <paramref name="type"/> rows of [N, K]. <paramref name="zeroPoints"/> are the packed 4-bit zero
    /// points (two per byte, ceil(blocks / 2) bytes per row), required for <see cref="DataType.Q4_1"/>.
    /// </summary>
    public static byte[] Pack(DataType type, byte[] codes, double[] scales, byte[]? zeroPoints, long n, long k, long bits, long blockSize)
    {
        long blocks = (k + blockSize - 1) / blockSize;
        long codeBytesPerBlock = blockSize * bits / 8;
        long zeroPointBytesPerRow = (blocks + 1) / 2;
        long subBlocks = k / BlockValues;
        int outBlockBytes = (int)type.ByteSize(BlockValues);
        var result = new byte[n * subBlocks * outBlockBytes];
        for (long r = 0; r < n; r++)
        {
            for (long sb = 0; sb < subBlocks; sb++)
            {
                long block = sb * BlockValues / blockSize;
                long first = sb * BlockValues % blockSize;
                float scale = (float)scales[r * blocks + block];
                var src = codes.AsSpan((int)((r * blocks + block) * codeBytesPerBlock + first * bits / 8), BlockValues * (int)bits / 8);
                var dst = result.AsSpan((int)((r * subBlocks + sb) * outBlockBytes), outBlockBytes);
                if (type == DataType.Q4_0)
                {
                    PackQ4(src, dst[2..]);
                    WriteHalf(dst, (Half)scale);
                }
                else if (type == DataType.Q4_1)
                {
                    int z = (zeroPoints![(int)(r * zeroPointBytesPerRow + block / 2)] >> (int)(4 * (block & 1))) & 0xF;
                    WriteHalf(dst, (Half)scale);
                    WriteHalf(dst[2..], (Half)(-z * scale));
                    PackQ4(src, dst[4..]);
                }
                else
                {
                    WriteHalf(dst, (Half)scale);
                    for (int j = 0; j < BlockValues; j++)
                    {
                        dst[2 + j] = (byte)(sbyte)(src[j] - 128);
                    }
                }
            }
        }

        return result;
    }

    /// <summary>MatMulNBits stores value j in nibble j % 2 of byte j / 2; the ggml formats put j and j + 16 in byte j.</summary>
    private static void PackQ4(ReadOnlySpan<byte> src, Span<byte> dst)
    {
        for (int j = 0; j < BlockValues / 2; j++)
        {
            int lo = Nibble(src, j);
            int hi = Nibble(src, j + BlockValues / 2);
            dst[j] = (byte)(lo | hi << 4);
        }
    }

    private static int Nibble(ReadOnlySpan<byte> src, int j) => (src[j / 2] >> (4 * (j & 1))) & 0xF;

    private static void WriteHalf(Span<byte> dst, Half value) => BitConverter.TryWriteBytes(dst, value);
}
