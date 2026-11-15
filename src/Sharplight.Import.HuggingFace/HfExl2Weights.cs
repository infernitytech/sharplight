using System.Buffers.Binary;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// ExLlamaV2 (EXL2) checkpoints: <c>X.q_weight</c> I32 [packed rows, N], <c>X.q_invperm</c> (or <c>X.q_perm</c>) [K], <c>X.q_scale</c>
/// I32 [groups, N / 8], <c>X.q_scale_max</c> F16 [groups] and <c>X.q_groups</c> I16 [groups, 2] become a dense F16 <c>X.weight</c>
/// [N, K]. EXL2 files carry no <c>quantization_config</c>; the modules are recognized by their tensors.
/// </summary>
/// <remarks>
/// Layout (exllamav2 <c>AdaptiveGPTQ.pack</c> and <c>pack_columns</c>, as stored before the loader's shuffle): the K rows are sorted into
/// groups whose bit widths (2, 3, 4, 5, 6 or 8) and first packed row are listed in <c>q_groups</c>; a group of R rows at b bits occupies
/// <c>R b / 32</c> packed rows, and per column its codes form one little-endian bit stream (code i at bit <c>i b</c>). The scale of a group
/// and column is the 4-bit value q of <c>q_scale</c> (eight per word, least significant first): <c>half((q + 1)^2) * half(q_scale_max / 256)</c>
/// rounded to f16, and a value is <c>half((code - 2^(b-1)) * scale)</c>, as exllamav2's <c>reconstruct</c> kernel computes it. Dense row
/// j of the transposed weight is stored row <c>q_invperm[j]</c>. This is a dequantize-at-import fallback: exllamav2 permutes activations
/// instead of weights, and group bit widths vary along K, which no block type expresses. Rows are decoded on demand.
/// </remarks>
internal static class HfExl2Weights
{
    private static readonly string[] s_suffixes = [".q_weight", ".q_invperm", ".q_perm", ".q_scale", ".q_scale_max", ".q_groups"];

    /// <summary>The checkpoint tensors with EXL2 modules replaced by dense weights; <paramref name="tensors"/> itself when there are none.</summary>
    /// <exception cref="NotSupportedException">A module has an unreadable layout.</exception>
    public static IReadOnlyDictionary<string, SafetensorsTensor> Resolve(IReadOnlyDictionary<string, SafetensorsTensor> tensors)
    {
        var modules = tensors.Keys.Where(name => name.EndsWith(".q_groups", StringComparison.Ordinal))
            .Select(name => name[..^".q_groups".Length])
            .Where(module => tensors.ContainsKey(module + ".q_weight") && tensors.ContainsKey(module + ".q_scale")
                && tensors.ContainsKey(module + ".q_scale_max"))
            .ToList();
        if (modules.Count == 0)
        {
            return tensors;
        }

        var result = new Dictionary<string, SafetensorsTensor>(tensors, StringComparer.Ordinal);
        foreach (string module in modules)
        {
            var weight = Convert(module, tensors);
            foreach (string suffix in s_suffixes)
            {
                result.Remove(module + suffix);
            }

            result[module + ".weight"] = weight;
        }

        return result;
    }

    private static unsafe SafetensorsTensor Convert(string module, IReadOnlyDictionary<string, SafetensorsTensor> tensors)
    {
        string name = module + ".weight";
        var codes = tensors[module + ".q_weight"];
        var scales = tensors[module + ".q_scale"];
        var maxima = tensors[module + ".q_scale_max"];
        var groupsTensor = tensors[module + ".q_groups"];
        bool inverse = tensors.TryGetValue(module + ".q_invperm", out var permutation);
        if (!inverse && !tensors.TryGetValue(module + ".q_perm", out permutation))
        {
            throw new NotSupportedException($"{name}: EXL2 module without q_invperm or q_perm.");
        }

        long columns = codes.Shape.Length == 2 ? codes.Shape[1] : 0;
        int groups = maxima.Shape.Length == 1 ? (int)maxima.Shape[0] : 0;
        int height = permutation!.Shape.Length == 1 ? (int)permutation.Shape[0] : 0;
        if (codes.DType != "I32" || columns == 0 || columns % 8 != 0 || scales.DType != "I32" || scales.Shape.Length != 2 || scales.Shape[0] != groups
            || scales.Shape[1] * 8 != columns || maxima.DType != "F16" || groupsTensor.DType != "I16" || groupsTensor.Shape.Aggregate(1L, (a, b) => a * b) != groups * 2L
            || height == 0 || permutation.DType is not ("I32" or "I16" or "I64"))
        {
            throw new NotSupportedException($"{name}: EXL2 tensors q_weight [{string.Join(", ", codes.Shape)}] {codes.DType}, q_scale [{string.Join(", ", scales.Shape)}], q_groups [{string.Join(", ", groupsTensor.Shape)}] are not a known layout.");
        }

        // Per stored K row: its group; per group: bits, first stored row, first packed row.
        var groupBits = new int[groups];
        var groupRow = new int[groups + 1];
        var groupPacked = new long[groups];
        int row = 0;
        for (int g = 0; g < groups; g++)
        {
            int bits = BinaryPrimitives.ReadInt16LittleEndian(new ReadOnlySpan<byte>(groupsTensor.Pointer + g * 4L, 2));
            int packedRow = BinaryPrimitives.ReadInt16LittleEndian(new ReadOnlySpan<byte>(groupsTensor.Pointer + g * 4L + 2, 2)) & 0xFFFF;
            if (bits is not (2 or 3 or 4 or 5 or 6 or 8))
            {
                throw new NotSupportedException($"{name}: EXL2 group of {bits} bits.");
            }

            groupBits[g] = bits;
            groupRow[g] = row;
            groupPacked[g] = packedRow;
            if (g < groups - 1)
            {
                int nextPacked = BinaryPrimitives.ReadInt16LittleEndian(new ReadOnlySpan<byte>(groupsTensor.Pointer + (g + 1) * 4L + 2, 2)) & 0xFFFF;
                row += (nextPacked - packedRow) * 32 / bits;
            }
            else
            {
                row = height;
            }
        }

        groupRow[groups] = height;
        long lastPackedEnd = groupPacked[groups - 1] + ((long)(height - groupRow[groups - 1]) * groupBits[groups - 1] + 31) / 32;
        if (row != height || groupRow[groups - 1] > height || lastPackedEnd > codes.Shape[0])
        {
            throw new NotSupportedException($"{name}: EXL2 q_groups do not cover the {height} rows of q_invperm.");
        }

        var rowGroup = new int[height];
        for (int g = 0; g < groups; g++)
        {
            Array.Fill(rowGroup, g, groupRow[g], groupRow[g + 1] - groupRow[g]);
        }

        // Dense column j reads stored row source[j].
        var source = new int[height];
        var seen = new bool[height];
        for (int i = 0; i < height; i++)
        {
            long value = permutation.DType switch
            {
                "I16" => BinaryPrimitives.ReadUInt16LittleEndian(new ReadOnlySpan<byte>(permutation.Pointer + i * 2L, 2)),
                "I32" => BinaryPrimitives.ReadInt32LittleEndian(new ReadOnlySpan<byte>(permutation.Pointer + i * 4L, 4)),
                _ => BinaryPrimitives.ReadInt64LittleEndian(new ReadOnlySpan<byte>(permutation.Pointer + i * 8L, 8)),
            };
            if (value < 0 || value >= height || seen[value])
            {
                throw new NotSupportedException($"{name}: EXL2 permutation is not a permutation of the {height} rows.");
            }

            seen[value] = true;
            if (inverse)
            {
                source[i] = (int)value;
            }
            else
            {
                source[value] = i;
            }
        }

        var decoder = new HfExl2Decoder(codes, scales, maxima, groupBits, groupRow, groupPacked, rowGroup, source, columns);
        var converted = new WeightInfo
        {
            Name = name,
            Type = DataType.F16,
            Shape = new Shape([(Dim)columns, (Dim)height]),
            Source = new HfRowConvertedSource(columns, height * 2L, decoder.DecodeRow),
        };
        return new SafetensorsTensor
        {
            Name = name,
            DType = "F16",
            Type = DataType.F16,
            Shape = [columns, height],
            Offset = codes.Offset,
            ByteLength = converted.ByteSize,
            File = codes.File,
            Converted = converted,
        };
    }
}
