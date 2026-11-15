using System.Buffers.Binary;
using System.Text.Json;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.HuggingFace;

/// <summary>
/// MLX affine-quantized checkpoints (<c>mlx.core.quantize</c>, as mlx-lm writes them: a <c>quantization</c> object
/// <c>{"group_size", "bits"}</c> without a method, optionally with per-module overrides): <c>X.weight</c> U32 [..., K * bits / 32]
/// holds the codes of each row as a little-endian bit stream, <c>X.scales</c> and <c>X.biases</c> [..., K / group_size] the group
/// affine parameters; value = scale * code + bias. 4-bit weights with groups of a multiple of 32 and scales and biases that are exact
/// f16 values become <see cref="DataType.Q4_1"/> (d = scale, m = bias per 32-value block), bit for bit; other bit widths and group
/// sizes decode to BF16.
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
        foreach (var (name, tensor) in tensors)
        {
            if (tensor.DType != "U32" || !name.EndsWith(".weight", StringComparison.Ordinal))
            {
                continue;
            }

            string stem = name[..^".weight".Length];
            if (!tensors.TryGetValue(stem + ".scales", out var scales) || !tensors.TryGetValue(stem + ".biases", out var biases))
            {
                continue;
            }

            var module = settings.TryGetProperty(stem, out var own) && own.ValueKind == JsonValueKind.Object ? own : settings;
            int group = HfPackedBits.Int(module, "group_size", HfPackedBits.Int(settings, "group_size", 64));
            converted[name] = Convert(tensor, scales, biases, group);
            consumed.Add(scales.Name);
            consumed.Add(biases.Name);
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

    /// <summary>The MLX quantization settings of a checkpoint, or null when it is not MLX-quantized.</summary>
    internal static JsonElement? Settings(HfConfig config)
    {
        foreach (string name in (string[])["quantization", "quantization_config"])
        {
            if (config.TryGetQuantization(name, out var settings)
                && !settings.TryGetProperty("quant_method", out _) && settings.TryGetProperty("bits", out var bits)
                && bits.ValueKind == JsonValueKind.Number && (HfPackedBits.String(settings, "mode") ?? "affine") == "affine")
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
        bool q41 = bits == 4 && group % 32 == 0 && HfPackedBits.FitsHalf(scales) && HfPackedBits.FitsHalf(biases);
        var type = q41 ? DataType.Q4_1 : DataType.BF16;
        Action<long, Span<byte>> convertRow = q41
            ? (row, destination) =>
            {
                byte* source = (byte*)packed + row * rowWordBytes;
                for (long b = 0; b < columns / 32; b++)
                {
                    long g = row * groups + b * 32 / group;
                    var block = destination.Slice((int)(b * 20), 20);
                    BinaryPrimitives.WriteHalfLittleEndian(block, (Half)HfPackedBits.Float(scales, g));
                    BinaryPrimitives.WriteHalfLittleEndian(block[2..], (Half)HfPackedBits.Float(biases, g));
                    byte* run = source + b * 16;
                    for (int j = 0; j < 16; j++)
                    {
                        int lo = Nibble(run, j);
                        int hi = Nibble(run, j + 16);
                        block[4 + j] = (byte)(lo | (hi << 4));
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

    /// <summary>Code <paramref name="index"/> of a 4-bit MLX run (the first of each pair in the low nibble).</summary>
    private static int Nibble(byte* run, int index) => (run[index >> 1] >> ((index & 1) * 4)) & 0xF;
}
