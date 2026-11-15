using System.Buffers;
using System.Buffers.Binary;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Packed integer weights whose groups carry an affine pair (value = scale * code + bias; <see cref="HfIntegerLayout.Affine"/>):
/// codes of up to 4 bits become <see cref="DataType.Q4_1"/> and 5-bit codes <see cref="DataType.Q5_1"/> (d = scale, m = bias in each
/// 32-value block of the group), bit for bit. Group sizes that are not a multiple of 32, scales that are not exactly f16 values and
/// wider codes decode to BF16. The codes are converted while they are read (<see cref="HfRowConvertedSource"/>); nothing is cached.
/// </summary>
internal static class HfAffineWeights
{
    /// <summary>The weight of <paramref name="layout"/> as block-quantized (or BF16) data.</summary>
    /// <param name="layout">An affine layout.</param>
    public static SafetensorsTensor Encode(HfIntegerLayout layout)
    {
        long columns = layout.Columns;
        int groups = checked((int)layout.Groups);
        bool block32 = layout.Bits <= 5 && layout.GroupSize % 32 == 0 && columns % 32 == 0 && layout.ScalesFitHalf;
        var type = !block32 ? DataType.BF16 : layout.Bits == 5 ? DataType.Q5_1 : DataType.Q4_1;
        int blockBytes = type.BlockBytes;
        long rowBytes = type.ByteSize(columns);

        void Convert(long row, Span<byte> destination)
        {
            byte[] codes = ArrayPool<byte>.Shared.Rent(checked((int)columns));
            float[] scales = ArrayPool<float>.Shared.Rent(groups);
            float[] biases = ArrayPool<float>.Shared.Rent(groups);
            try
            {
                layout.DecodeRow(row, codes.AsSpan(0, (int)columns), scales.AsSpan(0, groups), biases.AsSpan(0, groups));
                if (!block32)
                {
                    for (int k = 0; k < columns; k++)
                    {
                        int g = (int)(k / layout.GroupSize);
                        BinaryPrimitives.WriteUInt16LittleEndian(destination[(k * 2)..], HfFp8Codec.ToBFloat16(scales[g] * codes[k] + biases[g]));
                    }

                    return;
                }

                for (long b = 0; b < columns / 32; b++)
                {
                    var block = destination.Slice((int)(b * blockBytes), blockBytes);
                    int g = (int)(b * 32 / layout.GroupSize);
                    BinaryPrimitives.WriteHalfLittleEndian(block, (Half)scales[g]);
                    BinaryPrimitives.WriteHalfLittleEndian(block[2..], (Half)biases[g]);
                    uint high = 0;
                    int qsAt = layout.Bits == 5 ? 8 : 4;
                    for (int j = 0; j < 16; j++)
                    {
                        int lo = codes[b * 32 + j];
                        int hi = codes[b * 32 + j + 16];
                        block[qsAt + j] = (byte)((lo & 0xF) | ((hi & 0xF) << 4));
                        high |= (uint)(lo >> 4 & 1) << j | (uint)(hi >> 4 & 1) << (j + 16);
                    }

                    if (layout.Bits == 5)
                    {
                        BinaryPrimitives.WriteUInt32LittleEndian(block[4..], high);
                    }
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(codes);
                ArrayPool<float>.Shared.Return(scales);
                ArrayPool<float>.Shared.Return(biases);
            }
        }

        long[] shape = [layout.Rows, columns];
        var weight = new WeightInfo
        {
            Name = layout.Name,
            Type = type,
            Shape = new Shape(layout.Rows, columns),
            Source = new HfRowConvertedSource(layout.Rows, rowBytes, Convert),
        };
        return new SafetensorsTensor
        {
            Name = layout.Name,
            DType = layout.Origin.DType,
            Type = type,
            Shape = shape,
            Offset = layout.Origin.Offset,
            ByteLength = weight.ByteSize,
            File = layout.Origin.File,
            Converted = weight,
        };
    }
}
