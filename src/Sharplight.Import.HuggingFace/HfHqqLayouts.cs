using System.Text;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// HQQ checkpoints (<c>quantization_config.quant_method = "hqq"</c>; the layer state of hqq's <c>HQQLinear</c>: <c>X.W_q</c>, <c>X.scale</c>,
/// <c>X.zero</c>, <c>X.shape</c> [N, K], <c>X.nbits</c>, <c>X.group_size</c>, <c>X.axis</c>, <c>X.packing</c>, plus bookkeeping scalars).
/// value = (code - zero) * scale with a float zero point; the codes are bit-packed along the first dimension of the grouped matrix
/// (<c>8bit_u8</c>, <c>4bit_u8</c>, <c>2bit_u8</c>, <c>1bit_u8</c>, <c>3bit_32</c>: the unpacked row r is field r / S of packed row r % S, S the packed row count).
/// <list type="bullet">
/// <item><c>axis = 1</c>: the weight is cut into groups of consecutive values of the row-major [N, K] matrix (a group lies in one row when
/// the group size divides K): group, scale and zero are indexed by flat position / group size.</item>
/// <item><c>axis = 0</c>: the flat matrix is reshaped to [group size, N K / group size] and the groups are its columns: the group of a value is
/// its flat position modulo N K / group size, so every column of a row has its own group (a group size of 1 per row here, read as BF16).</item>
/// </list>
/// Scales and zero points that were themselves quantized (<c>quant_scale</c> / <c>quant_zero</c>) are not read.
/// </summary>
internal static unsafe class HfHqqLayouts
{
    private static readonly string[] s_companions =
    [
        "W_q", "scale", "zero", "shape", "nbits", "group_size", "axis", "packing", "unpack_view_dtype", "view_as_float", "quant_scale",
        "quant_zero", "compute_dtype", "offload_meta", "encoded_state_dict", "stores_quant_config", "channel_wise", "optimize", "round_zero",
    ];

    /// <summary>The layouts of the HQQ modules among <paramref name="tensors"/>; their tensors are added to <paramref name="consumed"/>.</summary>
    /// <exception cref="NotSupportedException">The module uses quantized scales / zero points or a packing that is not read.</exception>
    public static List<HfIntegerLayout> Find(IReadOnlyDictionary<string, SafetensorsTensor> tensors, HashSet<string> consumed)
    {
        var layouts = new List<HfIntegerLayout>();
        foreach (var (name, codes) in tensors)
        {
            if (!name.EndsWith(".W_q", StringComparison.Ordinal))
            {
                continue;
            }

            string stem = name[..^".W_q".Length];
            if (!tensors.TryGetValue(stem + ".scale", out var scales) || !tensors.TryGetValue(stem + ".zero", out var zeros)
                || !tensors.TryGetValue(stem + ".shape", out var shape) || shape.DType != "I64" || shape.Shape.Length != 1 || shape.Shape[0] != 2)
            {
                continue;
            }

            if (Flag(tensors, stem + ".quant_scale") || Flag(tensors, stem + ".quant_zero") || Flag(tensors, stem + ".view_as_float"))
            {
                throw new NotSupportedException($"{stem}: HQQ weights with quantized scales / zero points or float-viewed codes are not supported.");
            }

            long rows = ((long*)shape.Pointer)[0];
            long columns = ((long*)shape.Pointer)[1];
            int bits = Int(tensors, stem + ".nbits");
            int axis = Int(tensors, stem + ".axis");
            long group = Int(tensors, stem + ".group_size");
            string packing = tensors.TryGetValue(stem + ".packing", out var packingTensor) && packingTensor.ByteLength > 0
                ? Encoding.ASCII.GetString(packingTensor.Pointer, (int)packingTensor.ByteLength) : string.Empty;
            int fields = packing switch
            {
                "8bit_u8" => 1,
                "4bit_u8" => 2,
                "2bit_u8" => 4,
                "1bit_u8" => 8,
                "3bit_32" => 10,
                _ => throw new NotSupportedException($"{stem}: HQQ packing '{packing}' is not supported."),
            };
            if (bits is < 1 or > 8 || group <= 0 || (rows * columns) % group != 0 || (axis != 0 && axis != 1) || codes.Shape.Length != 2)
            {
                throw new NotSupportedException($"{stem}: HQQ weight with {bits} bits, group size {group} and axis {axis} is not supported.");
            }

            bool word = packing == "3bit_32";
            long packedRows = codes.Shape[0];
            long width = codes.Shape[1];
            long unpackedRows = axis == 1 ? rows * columns / group : group;
            long groupCount = rows * columns / group;
            int bitsPerField = word ? 3 : 8 / fields;
            if (width != (axis == 1 ? group : groupCount) || packedRows * fields < unpackedRows || (packedRows - 1) * fields >= unpackedRows + fields)
            {
                throw new NotSupportedException($"{stem}: HQQ codes of shape [{packedRows}, {width}] do not match the weight shape [{rows}, {columns}].");
            }

            nint raw = (nint)codes.Pointer;
            bool alongRow = axis == 1 && columns % group == 0;
            consumed.UnionWith(s_companions.Select(companion => stem + "." + companion).Where(tensors.ContainsKey));
            layouts.Add(new HfIntegerLayout
            {
                Name = stem + ".weight",
                Origin = codes,
                Bits = bits,
                Rows = rows,
                Columns = columns,
                GroupSize = alongRow ? group : 1,
                ScalesFitHalf = HfPackedBits.FitsHalf(scales) && HfPackedBits.FitsHalf(zeros),
                DecodeRow = (n, rowCodes, rowScales, rowZeros) =>
                {
                    for (long k = 0; k < columns; k++)
                    {
                        long flat = n * columns + k;
                        long unpackedRow = axis == 1 ? flat / group : flat / groupCount;
                        long column = axis == 1 ? flat % group : flat % groupCount;
                        long field = unpackedRow / packedRows;
                        long stored = unpackedRow % packedRows;
                        long position = stored * width + column;
                        int shift = bitsPerField * (fields - 1 - (int)field);
                        uint value = word ? ((uint*)raw)[position] : ((byte*)raw)[position];
                        rowCodes[(int)k] = (byte)((value >> shift) & ((1u << bitsPerField) - 1));
                    }

                    if (alongRow)
                    {
                        long perRow = columns / group;
                        for (long g = 0; g < perRow; g++)
                        {
                            rowScales[(int)g] = HfPackedBits.Float(scales, n * perRow + g);
                            rowZeros[(int)g] = HfPackedBits.Float(zeros, n * perRow + g);
                        }

                        return;
                    }

                    for (long k = 0; k < columns; k++)
                    {
                        long flat = n * columns + k;
                        long index = axis == 1 ? flat / group : flat % groupCount;
                        rowScales[(int)k] = HfPackedBits.Float(scales, index);
                        rowZeros[(int)k] = HfPackedBits.Float(zeros, index);
                    }
                },
            });
        }

        return layouts;
    }

    private static int Int(IReadOnlyDictionary<string, SafetensorsTensor> tensors, string name)
        => tensors.TryGetValue(name, out var tensor) && tensor.DType == "I32" ? *(int*)tensor.Pointer : throw new NotSupportedException($"{name} is missing.");

    private static bool Flag(IReadOnlyDictionary<string, SafetensorsTensor> tensors, string name)
        => tensors.TryGetValue(name, out var tensor) && tensor.ByteLength > 0 && *tensor.Pointer != 0;
}
