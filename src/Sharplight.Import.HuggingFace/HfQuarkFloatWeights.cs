using System.Buffers.Binary;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// AMD Quark's OCP microscaling weights besides the plain MXFP4 pack (<see cref="HfFp4Weights"/>), all U8 <c>X.weight</c> [N, ...] in blocks
/// of 32 values along K with a power-of-two scale:
/// <list type="bullet">
/// <item>MXFP6 (<c>dtype</c> <c>fp6_e3m2</c> / <c>fp6_e2m3</c>, 32-value groups, <c>scale_format</c> <c>e8m0</c>): <c>X.weight</c> [N, 3K / 4], four 6-bit codes in three
/// bytes (<c>Pack_fp6</c>: value 0 in bits 0 to 5 of the first byte, value 1 in its top two bits and the low four of the second,
/// value 2 in the top four of the second and the low two of the third, value 3 in the top six of the third), <c>X.weight_scale</c> U8 E8M0 [N, K / 32].</item>
/// <item>the in-stream variant (<c>dtype</c> <c>mx</c>, <c>mx_element_dtype</c> <c>fp4</c> / <c>fp6_e3m2</c> / <c>fp6_e2m3</c>; <c>Pack_mxfp4</c> / <c>Pack_mxfp6</c>): no scale tensor, every
/// block is its own scale (one signed byte, the base-2 exponent) followed by the codes: 16 bytes of FP4 pairs (low nibble first) or 24 bytes of
/// FP6 (per four values a byte of their top two bits, then two bytes of their low nibbles, first value in the high nibble).</item>
/// </list>
/// MXFP4 becomes <see cref="DataType.MXFP4"/> bit for bit. The library has no FP6 block type: MXFP6 becomes <see cref="DataType.F8E4M3_B32"/>, which
/// holds every FP6 value exactly (E3M2 and E2M3 values are E4M3 values), so the weights are as exact as the checkpoint's, in eight bits per value.
/// </summary>
internal static unsafe class HfQuarkFloatWeights
{
    private const int Block = 32;

    /// <summary>The checkpoint tensors with Quark's MXFP6 / in-stream MX weights converted and their scale tensors removed.</summary>
    /// <param name="config">The checkpoint config (its <c>quantization_config</c>).</param>
    /// <param name="tensors">The checkpoint tensors.</param>
    /// <exception cref="InvalidDataException">A weight's shape does not match its element format.</exception>
    public static IReadOnlyDictionary<string, SafetensorsTensor> Resolve(HfConfig config, IReadOnlyDictionary<string, SafetensorsTensor> tensors)
    {
        if (!config.TryGetQuantization("quantization_config", out var settings) || HfPackedBits.String(settings, "quant_method") != "quark")
        {
            return tensors;
        }

        var converted = new Dictionary<string, SafetensorsTensor>(StringComparer.Ordinal);
        var consumed = new HashSet<string>(StringComparer.Ordinal);
        foreach (var (name, weight) in tensors)
        {
            if (!name.EndsWith(".weight", StringComparison.Ordinal) || weight.DType != "U8" || weight.Shape.Length < 2
                || HfQuarkSpecs.WeightSpec(settings, name[..^".weight".Length]) is not { } spec)
            {
                continue;
            }

            string stem = name[..^".weight".Length];
            string dtype = HfPackedBits.String(spec, "dtype") ?? "";
            string element = dtype == "mx" ? HfPackedBits.String(spec, "mx_element_dtype") ?? "" : dtype;
            int mantissaBits = element switch { "fp6_e3m2" => 2, "fp6_e2m3" => 3, _ => 0 };
            bool embedded = dtype == "mx";
            if (!(mantissaBits != 0 || (embedded && element == "fp4")))
            {
                continue;
            }

            long packedBytes = embedded ? (element == "fp4" ? 17 : 25) : 24;
            long columns;
            SafetensorsTensor? scale = null;
            if (embedded)
            {
                columns = weight.Shape[^1] / packedBytes * Block;
                if (weight.Shape[^1] % packedBytes != 0)
                {
                    throw new InvalidDataException($"{name}: {weight.Shape[^1]} packed bytes per row are not a multiple of {packedBytes}.");
                }
            }
            else if (tensors.TryGetValue(stem + ".weight_scale", out scale) && scale.DType == "U8" && weight.Shape[^1] % 3 == 0)
            {
                columns = weight.Shape[^1] / 3 * 4;
                if (scale.Shape[^1] * Block != columns || !scale.Shape[..^1].SequenceEqual(weight.Shape[..^1]))
                {
                    throw new InvalidDataException($"{name}: scales [{string.Join(", ", scale.Shape)}] do not match the codes [{string.Join(", ", weight.Shape)}].");
                }

                consumed.Add(scale.Name);
            }
            else
            {
                continue;
            }

            consumed.Add(name);
            converted[name] = element == "fp4" ? Mxfp4(weight, columns) : Fp6(weight, scale, columns, mantissaBits);
        }

        if (converted.Count == 0)
        {
            return tensors;
        }

        return HfTensorMerge.Replace(tensors, consumed, converted);
    }

    /// <summary>The E4M3 code of every 6-bit code (sign, exponent, mantissa from the top) with the given mantissa width (2: E3M2, 3: E2M3).</summary>
    private static byte[] E4m3Codes(int mantissaBits)
    {
        int exponentBits = 5 - mantissaBits;
        int bias = (1 << (exponentBits - 1)) - 1;
        var codes = new byte[64];
        for (int code = 0; code < 64; code++)
        {
            int exponent = (code >> mantissaBits) & ((1 << exponentBits) - 1);
            int mantissa = code & ((1 << mantissaBits) - 1);
            float magnitude = exponent == 0
                ? MathF.ScaleB(mantissa, 1 - bias - mantissaBits)
                : MathF.ScaleB(1f + mantissa / (float)(1 << mantissaBits), exponent - bias);
            int match = -1;
            for (int e4m3 = 0; e4m3 < 127; e4m3++)
            {
                if (HfFp8Codec.E4m3((byte)e4m3) == magnitude)
                {
                    match = e4m3;
                    break;
                }
            }

            if (match < 0)
            {
                throw new InvalidOperationException($"FP6 value {magnitude} is not an E4M3 value.");
            }

            codes[code] = (byte)(match | ((code & 0x20) != 0 ? 0x80 : 0));
        }

        return codes;
    }

    private static SafetensorsTensor Fp6(SafetensorsTensor weight, SafetensorsTensor? scale, long columns, int mantissaBits)
    {
        byte[] table = E4m3Codes(mantissaBits);
        bool embedded = scale is null;
        long rowBytes = weight.Shape[^1];
        long rows = weight.Shape[..^1].Aggregate(1L, (a, d) => a * d);
        byte* codeBase = weight.Pointer;
        byte* scaleBase = scale?.Pointer;
        var type = DataType.F8E4M3_B32;
        return Result(weight, type, columns, rows, type.ByteSize(columns), (row, destination) =>
        {
            byte* packed = codeBase + row * rowBytes;
            for (long b = 0; b < columns / Block; b++)
            {
                var block = destination.Slice((int)(b * 36), 36);
                int exponent = embedded ? (sbyte)packed[b * 25] + 127 : scaleBase[row * (columns / Block) + b];
                BinaryPrimitives.WriteSingleLittleEndian(block, exponent is < 0 or > 254 ? (exponent > 254 ? float.NaN : 0f) : MathF.ScaleB(1f, exponent - 127));
                byte* data = packed + (embedded ? b * 25 + 1 : b * 24);
                for (int i = 0; i < Block; i++)
                {
                    int code = embedded ? EmbeddedCode(data, i) : PackedCode(data, i);
                    block[4 + i] = table[code];
                }
            }
        });
    }

    /// <summary>The 6-bit code <paramref name="i"/> of 32 in <c>Pack_fp6</c>'s stream (three bytes per four codes).</summary>
    private static int PackedCode(byte* data, int i)
    {
        byte* triple = data + i / 4 * 3;
        return (i % 4) switch
        {
            0 => triple[0] & 0x3F,
            1 => ((triple[1] & 0xF) << 2) | (triple[0] >> 6),
            2 => ((triple[2] & 3) << 4) | (triple[1] >> 4),
            _ => triple[2] >> 2,
        };
    }

    /// <summary>The 6-bit code <paramref name="i"/> of 32 in <c>Pack_mxfp6</c>'s stream (per four values: a byte of their top two bits, then two bytes of low nibbles).</summary>
    private static int EmbeddedCode(byte* data, int i)
    {
        byte* group = data + i / 4 * 3;
        int position = i % 4;
        int high = (group[0] >> (6 - 2 * position)) & 3;
        int low = position < 2 ? (group[1] >> (position == 0 ? 4 : 0)) & 0xF : (group[2] >> (position == 2 ? 4 : 0)) & 0xF;
        return (high << 4) | low;
    }

    private static SafetensorsTensor Mxfp4(SafetensorsTensor weight, long columns)
    {
        long rowBytes = weight.Shape[^1];
        long rows = weight.Shape[..^1].Aggregate(1L, (a, d) => a * d);
        byte* codeBase = weight.Pointer;
        return Result(weight, DataType.MXFP4, columns, rows, columns / Block * 17, (row, destination) =>
        {
            byte* packed = codeBase + row * rowBytes;
            for (long b = 0; b < columns / Block; b++)
            {
                var block = destination.Slice((int)(b * 17), 17);
                block[0] = (byte)Math.Clamp((sbyte)packed[b * 17] + 127, 0, 254);
                byte* run = packed + b * 17 + 1;
                for (int j = 0; j < 16; j++)
                {
                    block[1 + j] = (byte)(Nibble(run, j) | (Nibble(run, j + 16) << 4));
                }
            }
        });
    }

    private static int Nibble(byte* run, int index) => (run[index >> 1] >> ((index & 1) * 4)) & 0xF;

    private static SafetensorsTensor Result(SafetensorsTensor source, DataType type, long columns, long rows, long rowBytes, Action<long, Span<byte>> convertRow)
    {
        long[] shape = [.. source.Shape[..^1], columns];
        var weight = new WeightInfo
        {
            Name = source.Name,
            Type = type,
            Shape = new Shape(shape.Select(d => (Dim)d)),
            Source = new HfRowConvertedSource(rows, rowBytes, convertRow),
        };
        return new SafetensorsTensor
        {
            Name = source.Name,
            DType = source.DType,
            Type = type,
            Shape = shape,
            Offset = source.Offset,
            ByteLength = weight.ByteSize,
            File = source.File,
            Converted = weight,
        };
    }
}
