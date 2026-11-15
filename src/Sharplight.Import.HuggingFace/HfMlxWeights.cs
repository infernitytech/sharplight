using System.Buffers.Binary;
using System.Text.Json;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// MLX affine-quantized checkpoints (<c>mlx.core.quantize</c>, as mlx-lm writes them: a <c>quantization</c> object
/// <c>{"group_size", "bits"}</c> without a method, optionally with per-module overrides): <c>X.weight</c> U32 [..., K * bits / 32]
/// holds the codes of each row as a little-endian bit stream, <c>X.scales</c> and <c>X.biases</c> [..., K / group_size] the group
/// affine parameters; value = scale * code + bias. 2 to 4-bit weights (<see cref="DataType.Q4_1"/>) and 5-bit weights (<see cref="DataType.Q5_1"/>) with groups of a multiple of 32 and
/// scales and biases that are exact f16 values keep their codes (d = scale, m = bias per 32-value block), bit for bit; other bit
/// widths and group sizes decode to BF16. The FP modes (<c>mxfp4</c>, <c>nvfp4</c>, <c>mxfp8</c>, no biases) are read by
/// <see cref="HfMlxFloatWeights"/>.
/// </summary>
internal static unsafe class HfMlxWeights
{
    /// <summary>The checkpoint tensors with MLX-quantized weights converted and their scale and bias tensors removed.</summary>
    /// <param name="config">The checkpoint config (its <c>quantization</c> or method-less <c>quantization_config</c>).</param>
    /// <param name="tensors">The checkpoint tensors.</param>
    /// <exception cref="InvalidDataException">A weight's shape does not match its scales.</exception>
    public static IReadOnlyDictionary<string, SafetensorsTensor> Resolve(HfConfig config, IReadOnlyDictionary<string, SafetensorsTensor> tensors)
    {
        if (Settings(config) is not { } settings)
        {
            return tensors;
        }

        var converted = new Dictionary<string, SafetensorsTensor>(StringComparer.Ordinal);
        var consumed = new HashSet<string>(StringComparer.Ordinal);
        var floatCodes = new Dictionary<string, SafetensorsTensor>(StringComparer.Ordinal);
        foreach (var (name, tensor) in tensors)
        {
            if (tensor.DType != "U32" || !name.EndsWith(".weight", StringComparison.Ordinal))
            {
                continue;
            }

            string stem = name[..^".weight".Length];
            if (!tensors.TryGetValue(stem + ".scales", out var scales))
            {
                continue;
            }

            var module = settings.TryGetProperty(stem, out var own) && own.ValueKind == JsonValueKind.Object ? own : settings;
            string mode = HfPackedBits.String(module, "mode") ?? HfPackedBits.String(settings, "mode") ?? "affine";
            if (mode != "affine")
            {
                consumed.Add(scales.Name);
                HfMlxFloatWeights.Add(mode, stem, tensor, scales, floatCodes, converted);
                continue;
            }

            if (!tensors.TryGetValue(stem + ".biases", out var biases))
            {
                continue;
            }

            int group = HfPackedBits.Int(module, "group_size", HfPackedBits.Int(settings, "group_size", 64));
            converted[name] = Convert(tensor, scales, biases, group);
            consumed.Add(scales.Name);
            consumed.Add(biases.Name);
        }

        foreach (var (name, tensor) in HfFp4Weights.Resolve(floatCodes))
        {
            converted[name] = tensor;
        }

        if (converted.Count == 0)
        {
            return tensors;
        }

        var result = new Dictionary<string, SafetensorsTensor>(StringComparer.Ordinal);
        foreach (var (name, tensor) in tensors)
        {
            if (!consumed.Contains(name))
            {
                result[name] = converted.TryGetValue(name, out var weight) ? weight : tensor;
            }
        }

        return result;
    }

    /// <summary>The MLX quantization settings of a checkpoint (any <c>mode</c>), or null when it is not MLX-quantized.</summary>
    internal static JsonElement? Settings(HfConfig config)
    {
        foreach (string name in (string[])["quantization", "quantization_config"])
        {
            if (config.TryGetQuantization(name, out var settings)
                && !settings.TryGetProperty("quant_method", out _) && settings.TryGetProperty("bits", out var bits)
                && bits.ValueKind == JsonValueKind.Number)
            {
                return settings;
            }
        }

        return null;
    }

    private static SafetensorsTensor Convert(SafetensorsTensor codes, SafetensorsTensor scales, SafetensorsTensor biases, int group)
    {
        long groups = scales.Shape[^1];
        long columns = groups * group;
        long words = codes.Shape[^1];
        if (group <= 0 || words * 32 % columns != 0 || scales.Shape.Length != codes.Shape.Length)
        {
            throw new InvalidDataException($"{codes.Name}: {words} code words per row do not match {groups} groups of {group} values.");
        }

        int bits = checked((int)(words * 32 / columns));
        long[] shape = [.. codes.Shape[..^1], columns];
        long rows = shape.Aggregate(1L, (a, d) => a * d) / columns;
        nint packed = (nint)codes.Pointer;
        long rowWordBytes = words * 4;
        bool block32 = bits is >= 2 and <= 5 && group % 32 == 0 && HfPackedBits.FitsHalf(scales) && HfPackedBits.FitsHalf(biases);
        var type = !block32 ? DataType.BF16 : bits == 5 ? DataType.Q5_1 : DataType.Q4_1;
        int blockBytes = type.BlockBytes;
        Action<long, Span<byte>> convertRow = block32
            ? (row, destination) =>
            {
                byte* source = (byte*)packed + row * rowWordBytes;
                for (long b = 0; b < columns / 32; b++)
                {
                    long g = row * groups + b * 32 / group;
                    var block = destination.Slice((int)(b * blockBytes), blockBytes);
                    BinaryPrimitives.WriteHalfLittleEndian(block, (Half)HfPackedBits.Float(scales, g));
                    BinaryPrimitives.WriteHalfLittleEndian(block[2..], (Half)HfPackedBits.Float(biases, g));
                    // The 32 codes of the block, as Q4_1 / Q5_1 lay them out: value j in the low nibble and j + 16 in the high one of
                    // byte j; Q5_1 keeps the fifth bit of value i in bit i of a 32-bit mask in front of the nibbles.
                    uint high = 0;
                    int qsAt = bits == 5 ? 8 : 4;
                    for (int j = 0; j < 16; j++)
                    {
                        int lo = HfPackedBits.Code(source, 1, (b * 32 + j) * bits, bits);
                        int hi = HfPackedBits.Code(source, 1, (b * 32 + j + 16) * bits, bits);
                        block[qsAt + j] = (byte)((lo & 0xF) | ((hi & 0xF) << 4));
                        high |= (uint)(lo >> 4 & 1) << j | (uint)(hi >> 4 & 1) << (j + 16);
                    }

                    if (bits == 5)
                    {
                        BinaryPrimitives.WriteUInt32LittleEndian(block[4..], high);
                    }
                }
            }
            : (row, destination) =>
            {
                byte* source = (byte*)packed + row * rowWordBytes;
                for (long k = 0; k < columns; k++)
                {
                    long g = row * groups + k / group;
                    float value = HfPackedBits.Float(scales, g) * HfPackedBits.Code(source, 1, k * bits, bits) + HfPackedBits.Float(biases, g);
                    BinaryPrimitives.WriteUInt16LittleEndian(destination[(int)(k * 2)..], HfFp8Codec.ToBFloat16(value));
                }
            };

        var weight = new WeightInfo
        {
            Name = codes.Name,
            Type = type,
            Shape = new Shape(shape.Select(d => (Dim)d)),
            Source = new HfRowConvertedSource(rows, type.ByteSize(columns), convertRow),
        };
        return new SafetensorsTensor
        {
            Name = codes.Name,
            DType = codes.DType,
            Type = type,
            Shape = shape,
            Offset = codes.Offset,
            ByteLength = weight.ByteSize,
            File = codes.File,
            Converted = weight,
        };
    }
}
