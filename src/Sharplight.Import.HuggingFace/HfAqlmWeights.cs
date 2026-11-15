using System.Buffers.Binary;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// AQLM checkpoints (additive quantization with learned codebooks, <c>quant_method aqlm</c>): the tensors <c>X.codes</c>
/// [out groups, in groups, codebooks], <c>X.codebooks</c> [codebooks, codebook size, out group size, in group size] and
/// <c>X.scales</c> [out groups, 1, 1, 1] are folded into a dense <c>X.weight</c> while the weight is read.
/// </summary>
/// <remarks>
/// This is a dequantize-at-import fallback: the weight keeps the codebook dtype (F16 or BF16) instead of staying codebook-quantized,
/// because the repository has no codebook-sum decode kernel yet. The values equal the reference <c>aqlm</c> implementation
/// (<c>_dequantize_weight</c>: sum of the selected codebook vectors over the codebooks, then multiplied by the scale) bit for bit; rows
/// are decoded on demand from the memory-mapped tensors and nothing is cached.
/// </remarks>
internal static class HfAqlmWeights
{
    private const string CodesSuffix = ".codes";
    private const string CodebooksSuffix = ".codebooks";
    private const string ScalesSuffix = ".scales";

    /// <summary>The checkpoint tensors with AQLM modules replaced by dense weights; <paramref name="tensors"/> itself when there are none.</summary>
    /// <exception cref="NotSupportedException">A module has an unreadable layout (codebook dtype, code dtype, shapes).</exception>
    public static IReadOnlyDictionary<string, SafetensorsTensor> Resolve(IReadOnlyDictionary<string, SafetensorsTensor> tensors)
    {
        var modules = tensors.Keys.Where(name => name.EndsWith(CodesSuffix, StringComparison.Ordinal))
            .Select(name => name[..^CodesSuffix.Length])
            .Where(module => tensors.ContainsKey(module + CodebooksSuffix) && tensors.ContainsKey(module + ScalesSuffix))
            .ToList();
        if (modules.Count == 0)
        {
            return tensors;
        }

        var result = new Dictionary<string, SafetensorsTensor>(tensors, StringComparer.Ordinal);
        foreach (string module in modules)
        {
            var codes = tensors[module + CodesSuffix];
            var codebooks = tensors[module + CodebooksSuffix];
            var scales = tensors[module + ScalesSuffix];
            result.Remove(codes.Name);
            result.Remove(codebooks.Name);
            result.Remove(scales.Name);
            result[module + ".weight"] = Convert(module + ".weight", codes, codebooks, scales);
        }

        return result;
    }

    private static SafetensorsTensor Convert(string name, SafetensorsTensor codes, SafetensorsTensor codebooks, SafetensorsTensor scales)
    {
        if (codes.Shape.Length != 3 || codebooks.Shape.Length != 4 || scales.Shape.Length != 4 || codes.Shape[2] != codebooks.Shape[0]
            || scales.Shape[0] != codes.Shape[0] || scales.Shape[1] != 1 || scales.Shape[2] != 1 || scales.Shape[3] != 1)
        {
            throw new NotSupportedException($"{name}: AQLM tensors of shapes codes [{string.Join(", ", codes.Shape)}], codebooks [{string.Join(", ", codebooks.Shape)}], scales [{string.Join(", ", scales.Shape)}] are not a known layout.");
        }

        var codebookType = codebooks.DType switch
        {
            "F16" => DataType.F16,
            "BF16" => DataType.BF16,
            _ => throw new NotSupportedException($"{name}: AQLM codebooks of dtype {codebooks.DType} are not supported (F16 or BF16)."),
        };
        if (codes.DType is not ("I16" or "U8" or "I8" or "I32" or "U32") || scales.DType is not ("F16" or "BF16" or "F32"))
        {
            throw new NotSupportedException($"{name}: AQLM codes of dtype {codes.DType} / scales of dtype {scales.DType} are not supported.");
        }

        long codebookSize = codebooks.Shape[1];
        if (codebookSize < 2 || (codebookSize & (codebookSize - 1)) != 0)
        {
            throw new NotSupportedException($"{name}: AQLM codebook size {codebookSize} is not a power of two.");
        }

        long outGroups = codes.Shape[0];
        long inGroups = codes.Shape[1];
        long outGroupSize = codebooks.Shape[2];
        long inGroupSize = codebooks.Shape[3];
        var shape = new Shape([(Dim)(outGroups * outGroupSize), (Dim)(inGroups * inGroupSize)]);
        var source = new HfRowConvertedSource(outGroups * outGroupSize, codebookType.ByteSize(inGroups * inGroupSize),
            new Decoder(codes, codebooks, scales, codebookType, outGroupSize, inGroupSize).DecodeRow);
        var converted = new WeightInfo { Name = name, Type = codebookType, Shape = shape, Source = source };
        return new SafetensorsTensor
        {
            Name = name,
            DType = codebooks.DType,
            Type = codebookType,
            Shape = [outGroups * outGroupSize, inGroups * inGroupSize],
            Offset = codebooks.Offset,
            ByteLength = converted.ByteSize,
            File = codebooks.File,
            Converted = converted,
        };
    }

    private sealed unsafe class Decoder(SafetensorsTensor codes, SafetensorsTensor codebooks, SafetensorsTensor scales, DataType type, long outGroupSize, long inGroupSize)
    {
        private readonly long _inGroups = codes.Shape[1];
        private readonly long _codebookCount = codebooks.Shape[0];
        private readonly long _codebookSize = codebooks.Shape[1];

        /// <summary>Writes row <paramref name="row"/> of the dense weight (the reference sums in the codebook dtype, then scales).</summary>
        public void DecodeRow(long row, Span<byte> destination)
        {
            long outGroup = row / outGroupSize;
            long within = row % outGroupSize;
            float scale = Round(Load(scales, outGroup));
            long mask = _codebookSize - 1;
            for (long inGroup = 0; inGroup < _inGroups; inGroup++)
            {
                for (long b = 0; b < inGroupSize; b++)
                {
                    float sum = 0;
                    for (long c = 0; c < _codebookCount; c++)
                    {
                        long code = ReadCode((outGroup * _inGroups + inGroup) * _codebookCount + c) & mask;
                        long index = (((c * _codebookSize + code) * outGroupSize) + within) * inGroupSize + b;
                        sum += Load(codebooks, index);
                    }

                    Store(destination, (inGroup * inGroupSize + b), Round(sum) * scale);
                }
            }
        }

        private long ReadCode(long index) => codes.DType switch
        {
            "I16" => BinaryPrimitives.ReadInt16LittleEndian(new ReadOnlySpan<byte>(codes.Pointer + index * 2, 2)),
            "I8" => *(sbyte*)(codes.Pointer + index),
            "U8" => codes.Pointer[index],
            _ => BinaryPrimitives.ReadInt32LittleEndian(new ReadOnlySpan<byte>(codes.Pointer + index * 4, 4)),
        };

        private static float Load(SafetensorsTensor tensor, long index) => tensor.DType switch
        {
            "F16" => (float)BinaryPrimitives.ReadHalfLittleEndian(new ReadOnlySpan<byte>(tensor.Pointer + index * 2, 2)),
            "BF16" => BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadUInt16LittleEndian(new ReadOnlySpan<byte>(tensor.Pointer + index * 2, 2)) << 16),
            _ => BinaryPrimitives.ReadSingleLittleEndian(new ReadOnlySpan<byte>(tensor.Pointer + index * 4, 4)),
        };

        /// <summary>Rounds to the codebook dtype, as every tensor operation of the reference does.</summary>
        private float Round(float value)
        {
            if (type == DataType.F16)
            {
                return (float)(Half)value;
            }

            uint bits = BitConverter.SingleToUInt32Bits(value);
            if (float.IsNaN(value))
            {
                return value;
            }

            return BitConverter.UInt32BitsToSingle((bits + 0x7FFFu + ((bits >> 16) & 1)) & 0xFFFF0000u);
        }

        private void Store(Span<byte> destination, long index, float value)
        {
            if (type == DataType.F16)
            {
                BinaryPrimitives.WriteHalfLittleEndian(destination[(int)(index * 2)..], (Half)value);
                return;
            }

            BinaryPrimitives.WriteUInt16LittleEndian(destination[(int)(index * 2)..], (ushort)(BitConverter.SingleToUInt32Bits(Round(value)) >> 16));
        }
    }
}
