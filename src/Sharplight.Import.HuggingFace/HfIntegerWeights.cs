using System.Buffers;
using System.Buffers.Binary;
using System.Text.Json;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Packed integer checkpoints, the "integer codes with a group scale and zero point" family: GPTQ (auto-gptq / gptqmodel,
/// <see cref="HfGptqLayouts"/>), AWQ (<see cref="HfAwqLayouts"/>) and HQQ (<see cref="HfHqqLayouts"/>), EETQ (<see cref="HfEetqLayouts"/>) and compressed-tensors integer weights (<c>pack-quantized</c>,
/// <c>int-quantized</c>, <see cref="HfCompressedIntLayouts"/>). Each weight becomes one of the integer block types
/// <see cref="DataType.Q4Z_G128"/> / <see cref="DataType.Q4Z_G32"/> (up to 4 bits) or <see cref="DataType.Q8Z_G128"/> /
/// <see cref="DataType.Q8Z_G32"/> (5 to 8 bits), value = (code - zero) * scale with the checkpoint's f16 scale and zero point
/// repeated in each block of its group: the decoded values are those of the library's own dequantization, bit for bit.
/// </summary>
/// <remarks>
/// <list type="bullet">
/// <item>Block size: 128 when the group size (the row length for per-channel scales) is a multiple of 128, else 32 when it is a
/// multiple of 32. Other group sizes, and scales that are not exactly f16 values (BF16 / F32 scales are checked), decode to BF16.</item>
/// <item>Act-order (GPTQ <c>desc_act</c>, compressed-tensors <c>actorder: group</c>): the columns are stored sorted by group, so
/// groups are contiguous again, and the weight's <see cref="WeightInfo.InputOrder"/> makes the product gather its input in the same
/// order. Group indices that do not sort into equal contiguous groups decode to BF16.</item>
/// </list>
/// The codes are converted while they are read (<see cref="HfRowConvertedSource"/>); nothing is cached.
/// </remarks>
internal static class HfIntegerWeights
{
    /// <summary>The checkpoint tensors with packed integer weights converted and their companion tensors removed.</summary>
    /// <param name="config">The checkpoint config: its <c>quantization_config</c> names the packing.</param>
    /// <param name="tensors">The checkpoint tensors.</param>
    /// <exception cref="NotSupportedException">The packing variant is not read (e.g. AWQ <c>gemv_fast</c>).</exception>
    public static IReadOnlyDictionary<string, SafetensorsTensor> Resolve(HfConfig config, IReadOnlyDictionary<string, SafetensorsTensor> tensors)
    {
        var settings = config.TryGetQuantization("quantization_config", out var q) ? q : default;
        var consumed = new HashSet<string>(StringComparer.Ordinal);
        var layouts = new List<HfIntegerLayout>();
        string? method = settings.ValueKind == JsonValueKind.Object && settings.TryGetProperty("quant_method", out var m) ? m.GetString() : null;
        if (HfAutoRound.IsAutoRound(method))
        {
            method = HfAutoRound.Layout(settings) ?? throw new NotSupportedException(
                $"AutoRound packing '{HfPackedBits.String(settings, "packing_format")}' is not supported (auto_round:auto_gptq, auto_round:auto_awq).");
        }

        switch (method?.ToLowerInvariant())
        {
            case "gptq":
                layouts.AddRange(HfGptqLayouts.Find(settings, tensors, consumed));
                break;
            case "awq":
                layouts.AddRange(HfAwqLayouts.Find(settings, tensors, consumed));
                break;
            case "hqq":
                layouts.AddRange(HfHqqLayouts.Find(tensors, consumed));
                break;
            case "eetq":
                layouts.AddRange(HfEetqLayouts.Find(settings, tensors, consumed));
                break;
            case "compressed-tensors":
                layouts.AddRange(HfCompressedIntLayouts.Find(settings, tensors, consumed));
                break;
        }

        if (layouts.Count == 0)
        {
            return tensors;
        }

        var result = new Dictionary<string, SafetensorsTensor>(StringComparer.Ordinal);
        foreach (var (name, tensor) in tensors)
        {
            if (!consumed.Contains(name))
            {
                result[name] = tensor;
            }
        }

        foreach (var layout in layouts)
        {
            result[layout.Name] = Encode(layout);
        }

        return result;
    }

    /// <summary>The block type of a layout, or BF16 when its groups or scales fit none.</summary>
    internal static DataType TargetType(HfIntegerLayout layout)
    {
        if (!layout.ScalesFitHalf)
        {
            return DataType.BF16;
        }

        bool small = layout.Bits <= 4;
        if (layout.GroupSize % 128 == 0 && layout.Columns % 128 == 0)
        {
            return small ? DataType.Q4Z_G128 : DataType.Q8Z_G128;
        }

        return layout.GroupSize % 32 == 0 && layout.Columns % 32 == 0 ? (small ? DataType.Q4Z_G32 : DataType.Q8Z_G32) : DataType.BF16;
    }

    /// <summary>
    /// The stored column order of an act-order weight (columns sorted by group, stable), or null when the group indices are in
    /// natural order. When sorting does not make every group an equal contiguous run, returns null with <paramref name="sorted"/>
    /// false (the weight then decodes to BF16 in its stored order).
    /// </summary>
    internal static int[]? ColumnOrder(HfIntegerLayout layout, out bool sorted)
    {
        sorted = true;
        var groups = layout.GroupIndex;
        if (groups is null || Enumerable.Range(0, groups.Length).All(k => groups[k] == k / layout.GroupSize))
        {
            return null;
        }

        int[] order = [.. Enumerable.Range(0, groups.Length).OrderBy(k => groups[k])];
        for (int j = 0; j < order.Length; j++)
        {
            if (groups[order[j]] != j / layout.GroupSize)
            {
                sorted = false;
                return null;
            }
        }

        return order;
    }

    private static SafetensorsTensor Encode(HfIntegerLayout layout)
    {
        var order = ColumnOrder(layout, out bool sorted);
        var type = sorted ? TargetType(layout) : DataType.BF16;
        if (type == DataType.BF16)
        {
            order = null;
        }

        long columns = layout.Columns;
        int groups = checked((int)layout.Groups);
        int block = type.BlockSize;
        int blockBytes = type.BlockBytes;
        long rowBytes = type.ByteSize(columns);
        var groupIndex = layout.GroupIndex;
        void Convert(long row, Span<byte> destination)
        {
            byte[] codes = ArrayPool<byte>.Shared.Rent(checked((int)columns));
            float[] scales = ArrayPool<float>.Shared.Rent(groups);
            float[] zeros = ArrayPool<float>.Shared.Rent(groups);
            try
            {
                layout.DecodeRow(row, codes.AsSpan(0, (int)columns), scales.AsSpan(0, groups), zeros.AsSpan(0, groups));
                if (type == DataType.BF16)
                {
                    for (int k = 0; k < columns; k++)
                    {
                        int g = groupIndex is null ? (int)(k / layout.GroupSize) : groupIndex[k];
                        float scale = layout.ScalesFitHalf ? (float)(Half)scales[g] : scales[g];
                        BinaryPrimitives.WriteUInt16LittleEndian(destination[(k * 2)..], HfFp8Codec.ToBFloat16((codes[k] - zeros[g]) * scale));
                    }

                    return;
                }

                for (long b = 0; b < columns / block; b++)
                {
                    var target = destination.Slice((int)(b * blockBytes), blockBytes);
                    int g = (int)(b * block / layout.GroupSize);
                    BinaryPrimitives.WriteHalfLittleEndian(target, (Half)scales[g]);
                    BinaryPrimitives.WriteHalfLittleEndian(target[2..], (Half)zeros[g]);
                    long first = b * block;
                    if (block == blockBytes - 4)
                    {
                        for (int i = 0; i < block; i++)
                        {
                            target[4 + i] = codes[Column(first + i)];
                        }

                        continue;
                    }

                    for (int run = 0; run < block / 32; run++)
                    {
                        long start = first + 32 * run;
                        for (int i = 0; i < 16; i++)
                        {
                            target[4 + 16 * run + i] = (byte)(codes[Column(start + i)] | (codes[Column(start + i + 16)] << 4));
                        }
                    }
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(codes);
                ArrayPool<float>.Shared.Return(scales);
                ArrayPool<float>.Shared.Return(zeros);
            }
        }

        int Column(long stored) => order is null ? (int)stored : order[stored];

        long[] shape = [layout.Rows, columns];
        var weight = new WeightInfo
        {
            Name = layout.Name,
            Type = type,
            Shape = new Shape(layout.Rows, columns),
            Source = new HfRowConvertedSource(layout.Rows, rowBytes, Convert),
            InputOrder = order,
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
