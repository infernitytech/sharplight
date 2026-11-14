using System.Buffers.Binary;
using System.Text;
using System.Text.Json;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.HuggingFace;

/// <summary>
/// bitsandbytes checkpoints (transformers <c>BitsAndBytesConfig</c>, serialized weights):
/// <list type="bullet">
/// <item>4-bit (<c>load_in_4bit</c>): <c>X.weight</c> U8 [N * K / 2, 1], the row-major codes two per byte, the first in the high
/// nibble; <c>X.weight.absmax</c> one absmax per block of <c>blocksize</c> codes (F32, or U8 codes of a second quantization with
/// <c>X.weight.nested_absmax</c> per 256 blocks, <c>X.weight.nested_quant_map</c> and the offset of the quant state: absmax =
/// nested_quant_map[code] * nested_absmax[i / 256] + offset); <c>X.weight.quant_map</c> the 16-entry code book;
/// <c>X.weight.quant_state.bitsandbytes__nf4</c> (or <c>__fp4</c>) a JSON blob with the type, block size and shape. value =
/// quant_map[code] * absmax, in f32 as bitsandbytes computes it. NF4 weights become <see cref="DataType.NF4_B64"/> bit-exactly (the
/// f32 absmax repeated in each 64-value block of a larger block). FP4 code books and blocks that do not cover whole 64-value runs of
/// a row are decoded to BF16.</item>
/// <item>8-bit (LLM.int8, <c>load_in_8bit</c>): <c>X.weight</c> I8 [N, K] with <c>X.SCB</c> F32 [N]; value = q * SCB / 127. These
/// weights are decoded to BF16 (no block type carries an f32 row scale with 8-bit codes). The outlier threshold is an
/// inference-time split of the activations into an int8 product and an f16 product for the outlier columns; activations are not
/// quantized here, so every column takes the precise path bitsandbytes reserves for outliers.</item>
/// </list>
/// </summary>
internal static unsafe class HfBitsAndBytesWeights
{
    private const string QuantStateMarker = ".quant_state.bitsandbytes__";

    /// <summary>The checkpoint tensors with bitsandbytes weights converted and their quant-state tensors removed.</summary>
    /// <exception cref="InvalidDataException">A quant state is incomplete.</exception>
    public static IReadOnlyDictionary<string, SafetensorsTensor> Resolve(IReadOnlyDictionary<string, SafetensorsTensor> tensors)
    {
        var converted = new Dictionary<string, SafetensorsTensor>(StringComparer.Ordinal);
        var consumed = new HashSet<string>(StringComparer.Ordinal);
        foreach (var (name, tensor) in tensors)
        {
            int marker = name.IndexOf(QuantStateMarker, StringComparison.Ordinal);
            if (marker > 0)
            {
                string weightName = name[..marker];
                converted[weightName] = FourBit(weightName, tensor, tensors, consumed);
            }
            else if (name.EndsWith(".SCB", StringComparison.Ordinal) && tensors.TryGetValue(name[..^".SCB".Length] + ".weight", out var codes)
                     && codes.DType == "I8")
            {
                consumed.Add(name);
                converted[codes.Name] = EightBit(codes, tensor);
            }
        }

        if (converted.Count == 0)
        {
            return tensors;
        }

        var result = new Dictionary<string, SafetensorsTensor>(StringComparer.Ordinal);
        foreach (var (name, tensor) in tensors)
        {
            if (!consumed.Contains(name) && !converted.ContainsKey(name) && !name.EndsWith(".weight_format", StringComparison.Ordinal))
            {
                result[name] = tensor;
            }
        }

        foreach (var (name, tensor) in converted)
        {
            result[name] = tensor;
        }

        return result;
    }

    private static SafetensorsTensor FourBit(string name, SafetensorsTensor stateBlob, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HashSet<string> consumed)
    {
        using var state = JsonDocument.Parse(Encoding.UTF8.GetString(new ReadOnlySpan<byte>(stateBlob.Pointer, checked((int)stateBlob.ByteLength))));
        var root = state.RootElement;
        SafetensorsTensor Part(string suffix) => tensors.TryGetValue(name + suffix, out var part)
            ? part : throw new InvalidDataException($"{name}: bitsandbytes quant state without {name}{suffix}.");
        var codes = Part("");
        var absmax = Part(".absmax");
        var quantMap = Part(".quant_map");
        consumed.UnionWith([stateBlob.Name, absmax.Name, quantMap.Name]);
        long[] shape = [.. root.GetProperty("shape").EnumerateArray().Select(d => d.GetInt64())];
        int blocksize = root.GetProperty("blocksize").GetInt32();
        float[] codeBook = HfTransforms.DecodeF32(quantMap.ToWeightInfo());
        float[] scales;
        if (absmax.DType == "U8")
        {
            var nestedAbsmax = Part(".nested_absmax");
            var nestedMap = Part(".nested_quant_map");
            consumed.UnionWith([nestedAbsmax.Name, nestedMap.Name]);
            int nestedBlock = root.TryGetProperty("nested_blocksize", out var nb) ? nb.GetInt32() : 256;
            float offset = root.TryGetProperty("nested_offset", out var o) ? o.GetSingle() : 0f;
            float[] second = HfTransforms.DecodeF32(nestedAbsmax.ToWeightInfo());
            float[] secondMap = HfTransforms.DecodeF32(nestedMap.ToWeightInfo());
            scales = new float[absmax.ByteLength];
            for (long i = 0; i < scales.Length; i++)
            {
                scales[i] = secondMap[absmax.Pointer[i]] * second[i / nestedBlock] + offset;
            }
        }
        else
        {
            scales = HfTransforms.DecodeF32(absmax.ToWeightInfo());
        }

        long columns = shape[^1];
        long rows = shape.Aggregate(1L, (a, d) => a * d) / columns;
        nint packed = (nint)codes.Pointer;
        float Value(long index)
        {
            int code = (((byte*)packed)[index >> 1] >> ((index & 1) == 0 ? 4 : 0)) & 0xF;
            return codeBook[code] * scales[index / blocksize];
        }

        bool nf4 = codeBook.AsSpan().SequenceEqual(Nf4CodeBook);
        if (nf4 && columns % 64 == 0 && blocksize % 64 == 0)
        {
            return Result(name, codes, DataType.NF4_B64, shape, rows, columns / 64 * 36, (row, destination) =>
            {
                for (long b = 0; b < columns / 64; b++)
                {
                    long first = row * columns + b * 64;
                    var block = destination.Slice((int)(b * 36), 36);
                    BinaryPrimitives.WriteSingleLittleEndian(block, scales[first / blocksize]);
                    byte* source = (byte*)packed + first / 2;
                    for (int run = 0; run < 2; run++)
                    {
                        for (int i = 0; i < 16; i++)
                        {
                            int lo = Nibble(source, 32 * run + i);
                            int hi = Nibble(source, 32 * run + i + 16);
                            block[4 + 16 * run + i] = (byte)(lo | (hi << 4));
                        }
                    }
                }
            });
        }

        return Result(name, codes, DataType.BF16, shape, rows, columns * 2, (row, destination) =>
        {
            for (long k = 0; k < columns; k++)
            {
                BinaryPrimitives.WriteUInt16LittleEndian(destination[(int)(k * 2)..], HfFp8Codec.ToBFloat16(Value(row * columns + k)));
            }
        });
    }

    /// <summary>Code <paramref name="index"/> of a bitsandbytes run (the first of each pair in the high nibble).</summary>
    private static int Nibble(byte* run, int index) => (run[index >> 1] >> ((index & 1) == 0 ? 4 : 0)) & 0xF;

    /// <summary>The NF4 code book as bitsandbytes stores it (f32).</summary>
    private static ReadOnlySpan<float> Nf4CodeBook =>
    [
        -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f, -0.28444138169288635f, -0.18477343022823334f,
        -0.09105003625154495f, 0.0f, 0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
        0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f,
    ];

    private static SafetensorsTensor EightBit(SafetensorsTensor codes, SafetensorsTensor rowScales)
    {
        long rows = codes.Shape[0];
        long columns = codes.Shape[1];
        float[] scb = HfTransforms.DecodeF32(rowScales.ToWeightInfo());
        nint values = (nint)codes.Pointer;
        return Result(codes.Name, codes, DataType.BF16, codes.Shape, rows, columns * 2, (row, destination) =>
        {
            sbyte* source = (sbyte*)values + row * columns;
            for (long k = 0; k < columns; k++)
            {
                float value = source[k] * scb[row] * 7.874015718698502e-3f;
                BinaryPrimitives.WriteUInt16LittleEndian(destination[(int)(k * 2)..], HfFp8Codec.ToBFloat16(value));
            }
        });
    }

    private static SafetensorsTensor Result(string name, SafetensorsTensor origin, DataType type, long[] shape, long rows, long rowBytes,
        Action<long, Span<byte>> convertRow)
    {
        var weight = new WeightInfo
        {
            Name = name,
            Type = type,
            Shape = new Shape(shape.Select(d => (Dim)d)),
            Source = new HfRowConvertedSource(rows, rowBytes, convertRow),
        };
        return new SafetensorsTensor
        {
            Name = name,
            DType = origin.DType,
            Type = type,
            Shape = shape,
            Offset = origin.Offset,
            ByteLength = weight.ByteSize,
            File = origin.File,
            Converted = weight,
        };
    }
}
