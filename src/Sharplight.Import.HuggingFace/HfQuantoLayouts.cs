namespace Sharplight.Import.HuggingFace;

/// <summary>
/// optimum-quanto checkpoints (<c>QuantizedModelForCausalLM.save_pretrained</c>, <c>quantization_config.quant_method = "quanto"</c>; weights
/// <c>qint8</c>, <c>qint4</c>, <c>qint2</c>, <c>qfloat8</c>), tensors named after the quantized tensor's parts:
/// <list type="bullet">
/// <item><c>qint8</c>: <c>X.weight._data</c> I8 [N, K], <c>X.weight._scale</c> [N, 1]; value = data * scale. The signed codes become unsigned ones
/// with the zero point 128 (<see cref="HfIntegerWeights"/>).</item>
/// <item><c>qint4</c> / <c>qint2</c>: <c>X.weight._data._data</c> U8, <c>X.weight._scale</c> and <c>X.weight._shift</c> [R, 1] (R groups of
/// <c>group_size</c> consecutive values of the flat row-major [N, K] weight, 128 by default); value = scale * code - shift. The codes (4 or 2 bits,
/// chosen by the packed row count: R / rows) are packed along the first dimension of the grouped [R, group size] matrix: row r holds field
/// r / S of packed row r mod S (S the packed row count), field i in bits bits * i. They keep their codes as affine blocks
/// (<see cref="HfAffineWeights"/>, bias = -shift).</item>
/// <item><c>qfloat8</c>: <c>X.weight._data</c> F8_E4M3 [N, K] with <c>X.weight._scale</c>: renamed to <c>X.weight</c> / <c>X.weight_scale</c> for the FP8 path
/// (<see cref="HfFp8Weights"/>); the E5M2 variant is not read.</item>
/// </list>
/// The packed weights lose their [N, K] shape in the state dict (only the group matrix is stored; quanto takes the shape from the module it loads
/// into), so it is recovered from the product N * K and the model's hidden size H: the output-side projections (<c>o_proj</c>, <c>down_proj</c>,
/// <c>out_proj</c>, <c>fc2</c>, <c>w2</c>, <c>dense</c> ...) are [H, N * K / H], the others [N * K / H, H].
/// Activation scales (<c>input_scale</c>, <c>output_scale</c>) are dropped.
/// </summary>
internal static unsafe class HfQuantoLayouts
{
    private static readonly string[] s_outputSide = ["o_proj", "down_proj", "out_proj", "fc2", "w2", "wo", "c_proj", "dense", "proj_out", "dense_4h_to_h"];

    /// <summary>The checkpoint tensors with qfloat8 weights renamed to the FP8 scale layout (the others unchanged).</summary>
    /// <param name="tensors">The checkpoint tensors.</param>
    public static IReadOnlyDictionary<string, SafetensorsTensor> RenameFloat8(IReadOnlyDictionary<string, SafetensorsTensor> tensors)
    {
        if (!tensors.Any(pair => pair.Key.EndsWith(".weight._data", StringComparison.Ordinal) && pair.Value.DType is "F8_E4M3" or "F8_E5M2"))
        {
            return tensors;
        }

        var result = new Dictionary<string, SafetensorsTensor>(StringComparer.Ordinal);
        foreach (var (name, tensor) in tensors)
        {
            string? stem = name.EndsWith(".weight._data", StringComparison.Ordinal) ? name[..^".weight._data".Length] : null;
            if (stem is not null && tensor.DType is "F8_E4M3" or "F8_E5M2" && tensors.ContainsKey(stem + ".weight._scale"))
            {
                if (tensor.DType == "F8_E5M2")
                {
                    throw new NotSupportedException($"{stem}: quanto qfloat8 with E5M2 data is not supported (E4M3).");
                }

                result[stem + ".weight"] = Renamed(tensor, stem + ".weight");
            }
            else if (name.EndsWith(".weight._scale", StringComparison.Ordinal) && tensors.TryGetValue(name[..^"._scale".Length] + "._data", out var data)
                && data.DType == "F8_E4M3")
            {
                result[name[..^"._scale".Length] + "_scale"] = Renamed(tensor, name[..^"._scale".Length] + "_scale");
            }
            else
            {
                result[name] = tensor;
            }
        }

        return result;
    }

    internal static SafetensorsTensor Renamed(SafetensorsTensor tensor, string name) => new()
    {
        Name = name,
        DType = tensor.DType,
        Type = tensor.Type,
        Shape = tensor.Shape,
        Offset = tensor.Offset,
        ByteLength = tensor.ByteLength,
        File = tensor.File,
        Converted = tensor.Converted,
    };

    /// <summary>The layouts of the integer quanto modules among <paramref name="tensors"/>; their tensors are added to <paramref name="consumed"/>.</summary>
    /// <param name="config">The checkpoint config (the hidden size recovers the shapes of packed weights).</param>
    /// <param name="tensors">The checkpoint tensors.</param>
    /// <param name="consumed">Receives the tensors the layouts replace.</param>
    /// <exception cref="NotSupportedException">The weight shape cannot be recovered or the group layout is not read.</exception>
    public static List<HfIntegerLayout> Find(HfConfig config, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HashSet<string> consumed)
    {
        var layouts = new List<HfIntegerLayout>();
        foreach (var (name, data) in tensors)
        {
            if (name.EndsWith(".weight._data", StringComparison.Ordinal) && data.DType == "I8" && data.Shape.Length == 2)
            {
                string stem = name[..^".weight._data".Length];
                if (tensors.TryGetValue(stem + ".weight._scale", out var scale) && scale.Shape.Aggregate(1L, (a, d) => a * d) == data.Shape[0])
                {
                    consumed.UnionWith([name, scale.Name]);
                    layouts.Add(Int8(stem, data, scale));
                }
            }
            else if (name.EndsWith(".weight._data._data", StringComparison.Ordinal) && data.DType == "U8" && data.Shape.Length == 2)
            {
                string stem = name[..^".weight._data._data".Length];
                if (tensors.TryGetValue(stem + ".weight._scale", out var scale) && tensors.TryGetValue(stem + ".weight._shift", out var shift)
                    && shift.DType is "F16" or "BF16" or "F32")
                {
                    consumed.UnionWith([name, scale.Name, shift.Name]);
                    layouts.Add(SubByte(config, stem, data, scale, shift));
                }
            }
        }

        return layouts;
    }

    private static HfIntegerLayout Int8(string stem, SafetensorsTensor data, SafetensorsTensor scale)
    {
        long rows = data.Shape[0];
        long columns = data.Shape[1];
        nint codes = (nint)data.Pointer;
        return new HfIntegerLayout
        {
            Name = stem + ".weight",
            Origin = data,
            Bits = 8,
            Rows = rows,
            Columns = columns,
            GroupSize = columns,
            ScalesFitHalf = HfPackedBits.FitsHalf(scale),
            DecodeRow = (n, rowCodes, rowScales, rowZeros) =>
            {
                byte* row = (byte*)codes + n * columns;
                for (long k = 0; k < columns; k++)
                {
                    rowCodes[(int)k] = (byte)(row[k] ^ 0x80);   // the signed code plus 128
                }

                rowScales[0] = HfPackedBits.Float(scale, n);
                rowZeros[0] = 128;
            },
        };
    }

    private static HfIntegerLayout SubByte(HfConfig config, string stem, SafetensorsTensor data, SafetensorsTensor scale, SafetensorsTensor shift)
    {
        long packedRows = data.Shape[0];
        long group = data.Shape[1];
        long groupCount = scale.Shape.Aggregate(1L, (a, d) => a * d);
        long perByte = groupCount / packedRows;
        if (perByte is not (2 or 4) || (groupCount + perByte - 1) / perByte != packedRows || shift.Shape.Aggregate(1L, (a, d) => a * d) != groupCount)
        {
            throw new NotSupportedException($"{stem}: quanto weight of {groupCount} groups in {packedRows} packed rows is not a 4-bit or 2-bit pack.");
        }

        int bits = (int)(8 / perByte);
        long total = groupCount * group;
        int hidden = config.GetInt("hidden_size") ?? throw new NotSupportedException($"{stem}: the quanto weight shape needs the config's hidden_size.");
        if (total % hidden != 0)
        {
            throw new NotSupportedException($"{stem}: {total} weight values do not divide into the hidden size {hidden}.");
        }

        string module = stem[(stem.LastIndexOf('.') + 1)..];
        bool outputSide = s_outputSide.Contains(module);
        long rows = outputSide ? hidden : total / hidden;
        long columns = outputSide ? total / hidden : hidden;
        if (columns % group != 0)
        {
            throw new NotSupportedException($"{stem}: quanto groups of {group} values straddle the rows of a [{rows}, {columns}] weight.");
        }

        long perRow = columns / group;
        nint codes = (nint)data.Pointer;
        return new HfIntegerLayout
        {
            Name = stem + ".weight",
            Origin = data,
            Bits = bits,
            Rows = rows,
            Columns = columns,
            GroupSize = group,
            Affine = true,
            ScalesFitHalf = HfPackedBits.FitsHalf(scale) && HfPackedBits.FitsHalf(shift),
            DecodeRow = (n, rowCodes, rowScales, rowBiases) =>
            {
                byte* bytes = (byte*)codes;
                int mask = (1 << bits) - 1;
                for (long k = 0; k < columns; k++)
                {
                    long r = n * perRow + k / group;
                    long field = r / packedRows;
                    long stored = r % packedRows;
                    rowCodes[(int)k] = (byte)((bytes[stored * group + k % group] >> (int)(bits * field)) & mask);
                }

                for (long g = 0; g < perRow; g++)
                {
                    rowScales[(int)g] = HfPackedBits.Float(scale, n * perRow + g);
                    rowBiases[(int)g] = -HfPackedBits.Float(shift, n * perRow + g);
                }
            },
        };
    }
}
