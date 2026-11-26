using System.Buffers.Binary;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Packed FP4 (E2M1) checkpoints: NVFP4 from NVIDIA ModelOpt and compressed-tensors (<c>nvfp4-pack-quantized</c>), and MXFP4 from
/// compressed-tensors (<c>mxfp4-pack-quantized</c>). Their weights are turned into self-contained block-quantized weights before the
/// architecture mapping sees them, so every mapping, fused-projection split and expert stack works on them unchanged.
/// </summary>
/// <remarks>
/// <list type="bullet">
/// <item>Codes: <c>X.weight</c> (ModelOpt) or <c>X.weight_packed</c> (compressed-tensors), U8 [..., N, K/2], value 2i in the low
/// nibble of byte i and 2i+1 in the high one; E2M1 codes (bit 3 the sign, bits 0..2 index 0, 0.5, 1, 1.5, 2, 3, 4, 6).</item>
/// <item>NVFP4: <c>X.weight_scale</c> F8_E4M3 [..., N, K/16], one scale per 16 values, and a per-tensor f32 second-level scale,
/// <c>X.weight_scale_2</c> (ModelOpt, a multiplier), <c>X.weight_global_scale</c> (compressed-tensors, a divisor) or <c>X.weight_amax</c> /
/// <c>X.weight_outer_scale_denominator</c> (ModelOpt parameter groups, one factor per matrix; see <see cref="HfSuffixedScaleLayouts"/>). Value =
/// e2m1 * (e4m3(scale) * g), g = weight_scale_2 or 1 / weight_global_scale. When g is a power of two and every scaled E4M3 scale
/// is again an E4M3 value, g is folded into the scales and the weight becomes <see cref="DataType.NVFP4"/>; otherwise every
/// 64-value block keeps g in front: <see cref="DataType.NVFP4_GS"/> (5 bits per weight instead of 4.5). The ModelOpt multiplier
/// is applied exactly as ModelOpt dequantizes (scale product first); compressed-tensors divides the scale by its global scale,
/// which the reciprocal multiplier reproduces to within one f32 rounding.</item>
/// <item>MXFP4: <c>X.weight_scale</c> U8 (E8M0, bias 127) [..., N, K/32]; becomes <see cref="DataType.MXFP4"/> bit-exactly. DeepSeek's
/// own FP4 experts (DeepSeek-V4, <c>expert_dtype fp4</c>) use the same packing with I8 codes <c>X.weight</c> and an F8_E8M0
/// <c>X.scale</c>.</item>
/// <item>Static activation scales (<c>input_scale</c>, <c>input_global_scale</c>) are dropped: activations are not quantized.</item>
/// <item>NVFP4 rows whose K is not a multiple of 64 have no block type and are decoded to BF16.</item>
/// </list>
/// The codes are converted while they are read (<see cref="HfRowConvertedSource"/>); nothing is cached.
/// </remarks>
internal static class HfFp4Weights
{
    private const string ScaleSuffix = ".weight_scale";

    /// <summary>The scale suffix of DeepSeek's own FP4 experts (<c>X.scale</c>, F8_E8M0 only).</summary>
    private const string E8m0ScaleSuffix = ".scale";

    /// <summary>The checkpoint tensors with packed FP4 weights converted and their scale tensors removed.</summary>
    /// <exception cref="InvalidDataException">A packed FP4 weight's scale tensor does not match its shape.</exception>
    public static IReadOnlyDictionary<string, SafetensorsTensor> Resolve(IReadOnlyDictionary<string, SafetensorsTensor> tensors)
    {
        var converted = new Dictionary<string, SafetensorsTensor>(StringComparer.Ordinal);
        var consumed = new HashSet<string>(StringComparer.Ordinal);
        foreach (var (name, scale) in tensors)
        {
            // DeepSeek's own layout names the E8M0 scale X.scale and stores the packed codes as I8.
            bool deepSeek = name.EndsWith(E8m0ScaleSuffix, StringComparison.Ordinal) && scale.DType == SafetensorsTensor.E8m0DType;
            if (!name.EndsWith(ScaleSuffix, StringComparison.Ordinal) && !deepSeek)
            {
                continue;
            }

            string stem = name[..^(deepSeek ? E8m0ScaleSuffix : ScaleSuffix).Length];
            var codes = Codes(tensors, stem, deepSeek);
            bool nvfp4 = !deepSeek && scale.DType == "F8_E4M3";
            bool mxfp4 = scale.DType is "U8" or SafetensorsTensor.E8m0DType;
            if (codes is null || !(nvfp4 || mxfp4) || codes.Shape.Length < 2 || scale.Shape.Length != codes.Shape.Length)
            {
                continue;
            }

            int group = nvfp4 ? 16 : 32;
            long columns = codes.Shape[^1] * 2;
            if (scale.Shape[^1] * group != columns || !scale.Shape[..^1].SequenceEqual(codes.Shape[..^1]))
            {
                throw new InvalidDataException($"{name}: FP4 scales [{string.Join(", ", scale.Shape)}] do not match codes {codes.Name} [{string.Join(", ", codes.Shape)}].");
            }

            consumed.Add(codes.Name);
            consumed.Add(name);
            float[]? global = null;
            if (nvfp4)
            {
                global = GlobalScale(tensors, stem, consumed);
            }

            converted[stem + ".weight"] = nvfp4 ? Nvfp4(stem + ".weight", codes, scale, global) : Mxfp4(stem + ".weight", codes, scale);
        }

        if (converted.Count == 0)
        {
            return tensors;
        }

        return HfTensorMerge.Replace(tensors, consumed, converted);
    }

    private static SafetensorsTensor? Codes(IReadOnlyDictionary<string, SafetensorsTensor> tensors, string stem, bool signedBytes)
    {
        foreach (var suffix in new[] { ".weight_packed", ".weight" })
        {
            if (tensors.TryGetValue(stem + suffix, out var codes) && (codes.DType == "U8" || (signedBytes && codes.DType == "I8")))
            {
                return codes;
            }
        }

        return null;
    }

    /// <summary>The second-level multiplier g per matrix (one value, or one per leading index), 1 when the checkpoint has none.</summary>
    private static float[] GlobalScale(IReadOnlyDictionary<string, SafetensorsTensor> tensors, string stem, HashSet<string> consumed)
    {
        if (tensors.TryGetValue(stem + ".weight_scale_2", out var multiplier))
        {
            consumed.Add(multiplier.Name);
            return HfTransforms.DecodeF32(multiplier.ToWeightInfo());
        }

        if (tensors.TryGetValue(stem + ".weight_global_scale", out var divisor))
        {
            consumed.Add(divisor.Name);
            return [.. HfTransforms.DecodeF32(divisor.ToWeightInfo()).Select(v => 1f / v)];
        }

        // ModelOpt parameter groups (weight_scale_format "amax_denominator"): the factor of matrix e is amax[e] / outer_scale_denominator.
        if (tensors.TryGetValue(stem + ".weight_amax", out var amax) && tensors.TryGetValue(stem + ".weight_outer_scale_denominator", out var denominator))
        {
            consumed.Add(amax.Name);
            consumed.Add(denominator.Name);
            float d = HfTransforms.DecodeF32(denominator.ToWeightInfo())[0];
            return [.. HfTransforms.DecodeF32(amax.ToWeightInfo()).Select(v => v / d)];
        }

        return [1f];
    }

    private static unsafe SafetensorsTensor Nvfp4(string name, SafetensorsTensor codes, SafetensorsTensor scales, float[] global)
    {
        long columns = codes.Shape[^1] * 2;
        long rows = codes.Shape[^2];
        long matrices = codes.Shape[..^2].Aggregate(1L, (a, d) => a * d);
        if (global.Length != 1 && global.Length != matrices)
        {
            throw new InvalidDataException($"{name}: {global.Length} global scales for {matrices} matrices.");
        }

        byte* codeBase = codes.Pointer;
        byte* scaleBase = scales.Pointer;
        float Global(long row) => global.Length == 1 ? global[0] : global[row / rows];

        if (columns % 64 != 0)
        {
            return Result(name, codes, DataType.BF16, matrices * rows, columns * 2, (row, destination) =>
            {
                for (long k = 0; k < columns; k++)
                {
                    float multiplier = HfFp8Codec.E4m3(scaleBase[row * (columns / 16) + k / 16]) * Global(row);
                    float value = E2m1(Nibble(codeBase + row * (columns / 2), k)) * multiplier;
                    BinaryPrimitives.WriteUInt16LittleEndian(destination[(int)(k * 2)..], HfFp8Codec.ToBFloat16(value));
                }
            });
        }

        var folded = FoldedScaleCodes(global, scaleBase, matrices * rows * (columns / 16));
        var type = folded is null ? DataType.NVFP4_GS : DataType.NVFP4;
        int header = folded is null ? 4 : 0;
        int blockBytes = type.BlockBytes;
        return Result(name, codes, type, matrices * rows, columns / 64 * blockBytes, (row, destination) =>
        {
            byte* packed = codeBase + row * (columns / 2);
            byte* scale = scaleBase + row * (columns / 16);
            float g = Global(row);
            for (long b = 0; b < columns / 64; b++)
            {
                var block = destination.Slice((int)(b * blockBytes), blockBytes);
                if (folded is null)
                {
                    BinaryPrimitives.WriteSingleLittleEndian(block, g);
                }

                for (int s = 0; s < 4; s++)
                {
                    byte code = scale[b * 4 + s];
                    block[header + s] = folded is null ? code : folded[code];
                    byte* sub = packed + b * 32 + s * 8;
                    for (int i = 0; i < 8; i++)
                    {
                        block[header + 4 + 8 * s + i] = (byte)(Nibble(sub, i) | (Nibble(sub, i + 8) << 4));
                    }
                }
            }
        });
    }

    private static unsafe SafetensorsTensor Mxfp4(string name, SafetensorsTensor codes, SafetensorsTensor scales)
    {
        long columns = codes.Shape[^1] * 2;
        long rows = codes.Shape[..^1].Aggregate(1L, (a, d) => a * d);
        byte* codeBase = codes.Pointer;
        byte* scaleBase = scales.Pointer;
        return Result(name, codes, DataType.MXFP4, rows, columns / 32 * 17, (row, destination) =>
        {
            byte* packed = codeBase + row * (columns / 2);
            byte* scale = scaleBase + row * (columns / 32);
            for (long b = 0; b < columns / 32; b++)
            {
                var block = destination.Slice((int)(b * 17), 17);
                block[0] = scale[b];
                byte* run = packed + b * 16;
                for (int j = 0; j < 16; j++)
                {
                    block[1 + j] = (byte)(Nibble(run, j) | (Nibble(run, j + 16) << 4));
                }
            }
        });
    }

    /// <summary>Code <paramref name="index"/> of a packed run: the low nibble of byte index/2 for even indices, the high one for odd.</summary>
    private static unsafe int Nibble(byte* run, long index) => (run[index >> 1] >> (int)((index & 1) * 4)) & 0xF;

    private static float E2m1(int code)
    {
        float magnitude = (code & 7) switch { 0 => 0f, 1 => 0.5f, 2 => 1f, 3 => 1.5f, 4 => 2f, 5 => 3f, 6 => 4f, _ => 6f };
        return (code & 8) != 0 ? -magnitude : magnitude;
    }

    /// <summary>
    /// The E4M3 scale code of each original code once g is folded in (code(e4m3(c) * g)), when that is exact for all
    /// <paramref name="count"/> scales of the tensor: a single g that is a power of two and no scaled scale leaving the E4M3 values.
    /// Null otherwise.
    /// </summary>
    private static unsafe byte[]? FoldedScaleCodes(float[] global, byte* scales, long count)
    {
        float g = global[0];
        if (global.Length != 1 || !float.IsNormal(g) || (BitConverter.SingleToInt32Bits(g) & 0x7FFFFF) != 0)
        {
            return null;
        }

        var exact = new Dictionary<float, byte>();
        for (int code = 0; code < 0x7F; code++)
        {
            exact.TryAdd(HfFp8Codec.E4m3((byte)code), (byte)code);
        }

        var map = new byte[256];
        var valid = new bool[256];
        for (int code = 0; code < 0x7F; code++)
        {
            if (exact.TryGetValue(HfFp8Codec.E4m3((byte)code) * g, out byte scaled))
            {
                map[code] = scaled;
                valid[code] = true;
            }
        }

        for (long i = 0; i < count; i++)
        {
            if (!valid[scales[i]])
            {
                return null;
            }
        }

        return map;
    }

    private static SafetensorsTensor Result(string name, SafetensorsTensor codes, DataType type, long rows, long rowBytes, Action<long, Span<byte>> convertRow)
    {
        long[] shape = [.. codes.Shape[..^1], codes.Shape[^1] * 2];
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
