using System.Buffers.Binary;
using System.Text.Json;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// HIGGS checkpoints (IST-DASLab, <c>quant_method higgs</c>: Hadamard rotation, then vector quantization on a Gaussian-optimal grid,
/// stored in FLUTE's packed layout): <c>X.weight</c> I16 [N / 4, K], <c>X.scales</c> [N, K / group size], the grid <c>X.tables2</c>
/// [16, 16, 2] and the unused <c>X.tables</c> become a dense <c>X.weight</c> in the scales' dtype.
/// </summary>
/// <remarks>
/// <para>
/// Layout: FLUTE packs the code matrix W [K, N] (K = in features; the 8-bit grid index of an output row's input pair 2a, 2a + 1 is
/// split into the high nibble at row 2a and the low nibble at row 2a + 1) in tiles of the kernel template's <c>tileP</c> (32 or 64,
/// chosen by <c>tune_metadata[module].template_id</c>: 64 when <c>template_id % 48 &lt; 16</c> for 4 bits) and stores the int16 words
/// (four 4-bit values, least significant first) transposed. Output row n, input pair a reads word <c>[c, 2a + r]</c> with
/// <c>b = n / 4P, t = (n mod 4P) / P, p = n mod P, c = b P + p, r = t / 2</c>; the high nibble is value <c>2 (t mod 2) + 1</c>, the
/// low one <c>2 (t mod 2)</c>.
/// </para>
/// <para>
/// Values: the rotated row is <c>h[n, k] = grid[code][k mod 2] * scales[n, k / group size]</c>, and FLUTE multiplies activations by
/// the orthonormal Sylvester Hadamard matrix of <c>hadamard_size</c> before the GEMM, so the dense weight is that matrix applied to
/// each <c>hadamard_size</c> block of h (the matrix is symmetric). Activations are not rotated here. This is a dequantize-at-import
/// fallback: the repository has no grid-lookup kernel with an activation-side Hadamard op. Only the 4-bit, two-dimensional grid
/// (<c>bits 4, p 2</c>, the transformers default) is read; 2 and 3-bit and scalar grids are rejected. Rows are decoded on demand.
/// </para>
/// </remarks>
internal static class HfHiggsWeights
{
    /// <summary>The checkpoint tensors with HIGGS modules replaced by dense weights; <paramref name="tensors"/> itself when there are none.</summary>
    /// <exception cref="NotSupportedException">The configuration or a module's layout is not readable.</exception>
    public static IReadOnlyDictionary<string, SafetensorsTensor> Resolve(HfConfig config, IReadOnlyDictionary<string, SafetensorsTensor> tensors)
    {
        if (!config.TryGetQuantization("quantization_config", out var settings)
            || !settings.TryGetProperty("quant_method", out var method) || method.GetString() != "higgs")
        {
            return tensors;
        }

        var modules = tensors.Keys.Where(name => name.EndsWith(".tables2", StringComparison.Ordinal))
            .Select(name => name[..^".tables2".Length])
            .Where(module => tensors.TryGetValue(module + ".weight", out var weight) && weight.DType == "I16" && tensors.ContainsKey(module + ".scales"))
            .ToList();
        if (modules.Count == 0)
        {
            return tensors;
        }

        int bits = HfPackedBits.Int(settings, "bits", 4);
        int dimensions = HfPackedBits.Int(settings, "p", 2);
        int hadamard = HfPackedBits.Int(settings, "hadamard_size", 512);
        if (bits != 4 || dimensions != 2)
        {
            throw new NotSupportedException($"HIGGS with {bits} bits and {dimensions}-dimensional grids is not supported (4 bits, p = 2).");
        }

        if (hadamard < 2 || (hadamard & (hadamard - 1)) != 0)
        {
            throw new NotSupportedException($"HIGGS hadamard_size {hadamard} is not a power of two.");
        }

        if (!settings.TryGetProperty("tune_metadata", out var metadata) || metadata.ValueKind != JsonValueKind.Object)
        {
            throw new NotSupportedException("HIGGS checkpoint without tune_metadata: the FLUTE tile layout of its weights is unknown.");
        }

        var result = new Dictionary<string, SafetensorsTensor>(tensors, StringComparer.Ordinal);
        foreach (string module in modules)
        {
            if (!metadata.TryGetProperty(module, out var entry) || !entry.TryGetProperty("template_id", out var template))
            {
                throw new NotSupportedException($"{module}: HIGGS module without a template_id in tune_metadata.");
            }

            int tileP = template.GetInt32() % 48 < 16 ? 64 : 32;
            var codes = tensors[module + ".weight"];
            var scales = tensors[module + ".scales"];
            var grid = tensors[module + ".tables2"];
            result.Remove(module + ".tables");
            result.Remove(module + ".tables2");
            result.Remove(module + ".scales");
            result[module + ".weight"] = Convert(module + ".weight", codes, scales, grid, tileP, hadamard);
        }

        return result;
    }

    private static SafetensorsTensor Convert(string name, SafetensorsTensor codes, SafetensorsTensor scales, SafetensorsTensor grid, int tileP, int hadamard)
    {
        long rows = codes.Shape.Length == 2 ? codes.Shape[0] * 4 : 0;
        long columns = codes.Shape.Length == 2 ? codes.Shape[1] : 0;
        if (rows == 0 || scales.Shape.Length != 2 || scales.Shape[0] != rows || columns % scales.Shape[1] != 0
            || grid.Shape.Length != 3 || grid.Shape[0] != 16 || grid.Shape[1] != 16 || grid.Shape[2] != 2 || rows % (4 * tileP) != 0
            || columns % hadamard != 0 || columns % 2 != 0)
        {
            throw new NotSupportedException($"{name}: HIGGS tensors of shapes weight [{string.Join(", ", codes.Shape)}], scales [{string.Join(", ", scales.Shape)}], tables2 [{string.Join(", ", grid.Shape)}] are not a known layout.");
        }

        var type = scales.DType switch
        {
            "F16" => DataType.F16,
            "BF16" => DataType.BF16,
            _ => throw new NotSupportedException($"{name}: HIGGS scales of dtype {scales.DType} are not supported (F16 or BF16)."),
        };
        if (grid.DType != scales.DType)
        {
            throw new NotSupportedException($"{name}: HIGGS grid dtype {grid.DType} differs from the scales' {scales.DType}.");
        }

        var values = HfTransforms.DecodeF32(grid.ToWeightInfo());
        var decoder = new Decoder(codes, scales, values, type, tileP, hadamard, columns, columns / scales.Shape[1]);
        var converted = new WeightInfo
        {
            Name = name,
            Type = type,
            Shape = new Shape([(Dim)rows, (Dim)columns]),
            Source = new HfRowConvertedSource(rows, type.ByteSize(columns), decoder.DecodeRow),
        };
        return new SafetensorsTensor
        {
            Name = name,
            DType = scales.DType,
            Type = type,
            Shape = [rows, columns],
            Offset = codes.Offset,
            ByteLength = converted.ByteSize,
            File = codes.File,
            Converted = converted,
        };
    }

    private sealed unsafe class Decoder(SafetensorsTensor codes, SafetensorsTensor scales, float[] grid, DataType type, int tileP, int hadamard, long columns, long group)
    {
        public void DecodeRow(long row, Span<byte> destination)
        {
            long b = row / (4 * tileP);
            int t = (int)(row % (4 * tileP) / tileP);
            long word = b * tileP + row % tileP;
            int r = t >> 1;
            int highShift = 4 * (2 * (t & 1) + 1);
            int lowShift = 4 * (2 * (t & 1));
            byte* packed = codes.Pointer + word * columns * 2;
            byte* scaleRow = scales.Pointer + row * (columns / group) * 2;
            var values = new float[columns];
            for (long a = 0; a < columns / 2; a++)
            {
                int bitsWord = BinaryPrimitives.ReadUInt16LittleEndian(new ReadOnlySpan<byte>(packed + (2 * a + r) * 2, 2));
                int code = (((bitsWord >> highShift) & 15) << 4) | ((bitsWord >> lowShift) & 15);
                float scale = Scale(scaleRow, 2 * a / group);
                values[2 * a] = grid[2 * code] * scale;
                values[2 * a + 1] = grid[2 * code + 1] * scale;
            }

            float norm = 1f / MathF.Sqrt(hadamard);
            for (long start = 0; start < columns; start += hadamard)
            {
                var block = values.AsSpan((int)start, hadamard);
                for (int half = 1; half < hadamard; half <<= 1)
                {
                    for (int i = 0; i < hadamard; i += 2 * half)
                    {
                        for (int j = i; j < i + half; j++)
                        {
                            float x = block[j];
                            float y = block[j + half];
                            block[j] = x + y;
                            block[j + half] = x - y;
                        }
                    }
                }

                for (int i = 0; i < hadamard; i++)
                {
                    block[i] *= norm;
                }
            }

            for (long i = 0; i < columns; i++)
            {
                if (type == DataType.F16)
                {
                    BinaryPrimitives.WriteHalfLittleEndian(destination[(int)(i * 2)..], (Half)values[i]);
                }
                else
                {
                    BinaryPrimitives.WriteUInt16LittleEndian(destination[(int)(i * 2)..], ToBf16(values[i]));
                }
            }
        }

        private float Scale(byte* scaleRow, long index)
        {
            ushort bits = BinaryPrimitives.ReadUInt16LittleEndian(new ReadOnlySpan<byte>(scaleRow + index * 2, 2));
            return type == DataType.F16 ? (float)BitConverter.UInt16BitsToHalf(bits) : BitConverter.Int32BitsToSingle(bits << 16);
        }

        private static ushort ToBf16(float value)
        {
            if (float.IsNaN(value))
            {
                return 0x7FC0;
            }

            uint bits = BitConverter.SingleToUInt32Bits(value);
            return (ushort)((bits + 0x7FFFu + ((bits >> 16) & 1)) >> 16);
        }
    }
}
