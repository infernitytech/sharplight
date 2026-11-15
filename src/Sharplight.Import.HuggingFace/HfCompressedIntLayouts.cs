using System.Text.Json;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// compressed-tensors integer weights (llm-compressor; <c>quantization_config.quant_method = "compressed-tensors"</c>), value =
/// (q - zero_point) * scale with signed q:
/// <list type="bullet">
/// <item><c>pack-quantized</c>: <c>X.weight_packed</c> I32 [N, ceil(K / (32 / bits))], 32 / bits codes per word along K (code i of a
/// word in bits [bits * i, bits * (i + 1)), stored as q + 2^(bits-1)); <c>X.weight_shape</c> [2]; <c>X.weight_scale</c> [N, groups]
/// (one group per row for channel scales, [1] for a tensor scale); for asymmetric schemes <c>X.weight_zero_point</c> I32
/// [ceil(N / (32 / bits)), groups], packed the same way along N; for act-order (<c>actorder: group</c>) <c>X.weight_g_idx</c> I32 [K].</item>
/// <item><c>int-quantized</c> / <c>naive-quantized</c>: <c>X.weight</c> I8 [N, K] with <c>X.weight_scale</c> and an optional I8
/// <c>X.weight_zero_point</c>.</item>
/// </list>
/// In unsigned terms the zero point is the stored zero point code (or 2^(bits-1) without one), so the weight maps to the integer
/// block types exactly.
/// </summary>
internal static unsafe class HfCompressedIntLayouts
{
    /// <summary>
    /// The layouts of the integer modules among <paramref name="tensors"/>; their tensors are added to <paramref name="consumed"/>.
    /// Modules in other compressed-tensors formats (FP8, FP4) are left to their converters.
    /// </summary>
    public static List<HfIntegerLayout> Find(JsonElement settings, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HashSet<string> consumed)
    {
        int bits = WeightBits(settings);
        var layouts = new List<HfIntegerLayout>();
        foreach (var (name, tensor) in tensors)
        {
            if (name.EndsWith(".weight_packed", StringComparison.Ordinal) && tensor.DType == "I32" && tensor.Shape.Length == 2)
            {
                string stem = name[..^".weight_packed".Length];
                if (tensors.TryGetValue(stem + ".weight_scale", out var scale))
                {
                    layouts.Add(Packed(stem, tensor, scale, tensors, bits, consumed));
                }
            }
            else if (name.EndsWith(".weight", StringComparison.Ordinal) && tensor.DType == "I8" && tensor.Shape.Length == 2)
            {
                string stem = name[..^".weight".Length];
                if (tensors.TryGetValue(stem + ".weight_scale", out var scale))
                {
                    layouts.Add(Bytes(stem, tensor, scale, tensors, consumed));
                }
            }
        }

        return layouts;
    }

    /// <summary>The integer weight width of the config groups (4 when none names one).</summary>
    private static int WeightBits(JsonElement settings)
    {
        if (settings.TryGetProperty("config_groups", out var groups) && groups.ValueKind == JsonValueKind.Object)
        {
            foreach (var group in groups.EnumerateObject())
            {
                if (group.Value.ValueKind == JsonValueKind.Object && group.Value.TryGetProperty("weights", out var weights)
                    && HfPackedBits.String(weights, "type") is null or "int")
                {
                    return HfPackedBits.Int(weights, "num_bits", 4);
                }
            }
        }

        return 4;
    }

    private static HfIntegerLayout Packed(string stem, SafetensorsTensor packed, SafetensorsTensor scale,
        IReadOnlyDictionary<string, SafetensorsTensor> tensors, int bits, HashSet<string> consumed)
    {
        int perWord = 32 / bits;
        long rows = packed.Shape[0];
        long columns = packed.Shape[1] * perWord;
        if (tensors.TryGetValue(stem + ".weight_shape", out var shape) && HfTransforms.DecodeF32(shape.ToWeightInfo()) is { Length: 2 } dims)
        {
            columns = (long)dims[1];
        }

        tensors.TryGetValue(stem + ".weight_zero_point", out var zeroPoint);
        tensors.TryGetValue(stem + ".weight_g_idx", out var gIdx);
        consumed.UnionWith(new[] { packed, scale, shape, zeroPoint, gIdx }.OfType<SafetensorsTensor>().Select(t => t.Name));
        long groups = scale.Shape.Length == 2 ? scale.Shape[1] : 1;
        bool perTensor = scale.Shape.Aggregate(1L, (a, d) => a * d) == 1;
        nint words = (nint)packed.Pointer;
        long rowWords = packed.Shape[1];
        nint zeroWords = zeroPoint is null ? 0 : (nint)zeroPoint.Pointer;
        int offset = 1 << (bits - 1);
        return new HfIntegerLayout
        {
            Name = stem + ".weight",
            Origin = packed,
            Bits = bits,
            Rows = rows,
            Columns = columns,
            GroupSize = (columns + groups - 1) / groups,
            GroupIndex = gIdx is null ? null : HfPackedBits.Int32s(gIdx),
            ScalesFitHalf = HfPackedBits.FitsHalf(scale),
            DecodeRow = (n, rowCodes, rowScales, rowZeros) =>
            {
                uint* row = (uint*)words + n * rowWords;
                for (long k = 0; k < columns; k++)
                {
                    rowCodes[(int)k] = (byte)((row[k / perWord] >> (int)(bits * (k % perWord))) & ((1u << bits) - 1));
                }

                for (long g = 0; g < groups; g++)
                {
                    rowScales[(int)g] = HfPackedBits.Float(scale, perTensor ? 0 : n * groups + g);
                    rowZeros[(int)g] = zeroWords == 0
                        ? offset
                        : (*((uint*)zeroWords + n / perWord * groups + g) >> (int)(bits * (n % perWord))) & ((1u << bits) - 1);
                }
            },
        };
    }

    private static HfIntegerLayout Bytes(string stem, SafetensorsTensor weight, SafetensorsTensor scale,
        IReadOnlyDictionary<string, SafetensorsTensor> tensors, HashSet<string> consumed)
    {
        tensors.TryGetValue(stem + ".weight_zero_point", out var zeroPoint);
        consumed.UnionWith(new[] { weight, scale, zeroPoint }.OfType<SafetensorsTensor>().Select(t => t.Name));
        long rows = weight.Shape[0];
        long columns = weight.Shape[1];
        long groups = scale.Shape.Length == 2 ? scale.Shape[1] : 1;
        bool perTensor = scale.Shape.Aggregate(1L, (a, d) => a * d) == 1;
        nint codes = (nint)weight.Pointer;
        nint zeros = zeroPoint is null ? 0 : (nint)zeroPoint.Pointer;
        bool oneZero = zeroPoint is not null && zeroPoint.Shape.Aggregate(1L, (a, d) => a * d) == 1;
        return new HfIntegerLayout
        {
            Name = stem + ".weight",
            Origin = weight,
            Bits = 8,
            Rows = rows,
            Columns = columns,
            GroupSize = (columns + groups - 1) / groups,
            ScalesFitHalf = HfPackedBits.FitsHalf(scale),
            DecodeRow = (n, rowCodes, rowScales, rowZeros) =>
            {
                sbyte* row = (sbyte*)codes + n * columns;
                for (long k = 0; k < columns; k++)
                {
                    rowCodes[(int)k] = (byte)(row[k] + 128);
                }

                for (long g = 0; g < groups; g++)
                {
                    long index = perTensor ? 0 : n * groups + g;
                    rowScales[(int)g] = HfPackedBits.Float(scale, index);
                    rowZeros[(int)g] = 128 + (zeros == 0 ? 0 : ((sbyte*)zeros)[oneZero ? 0 : index]);
                }
            },
        };
    }
}
