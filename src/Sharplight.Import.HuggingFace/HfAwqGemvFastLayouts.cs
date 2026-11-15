using System.Text.Json;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// AWQ <c>gemv_fast</c> checkpoints (AutoAWQ <c>WQLinear_GEMVFast</c>, 4-bit codes): <c>X.qweight</c> I16 [N / 4, K], four output rows
/// interleaved per word (nibble t of a word holds row 4 * (N / 4 index) + t, as AutoAWQ's <c>pack_intweight</c> lays it out: each
/// 64-column block of the four rows is the flat sequence row 0, row 1, row 2, row 3 of 64 codes cut into words of four, and the
/// columns inside every 32 are permuted for fast dequantization); <c>X.scales</c> F16 [groups padded, N] and <c>X.qzeros</c> F16
/// [groups padded, N], which holds -scale * zero, so value = scale * code + qzeros (<see cref="HfIntegerLayout.Affine"/>).
/// </summary>
internal static unsafe class HfAwqGemvFastLayouts
{
    /// <summary>Column inside a 32-block of the code stored at each position (inverse of the two fast-dequantization permutations).</summary>
    private static readonly int[] s_column = BuildColumnOrder();

    /// <summary>The layouts of the <c>gemv_fast</c> modules among <paramref name="tensors"/>; their tensors are added to <paramref name="consumed"/>.</summary>
    /// <exception cref="NotSupportedException">The bit width is not 4.</exception>
    public static List<HfIntegerLayout> Find(JsonElement settings, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HashSet<string> consumed)
    {
        int bits = HfPackedBits.Int(settings, "bits", HfPackedBits.Int(settings, "w_bit", 4));
        if (bits != 4)
        {
            throw new NotSupportedException($"AWQ gemv_fast with {bits}-bit codes is not supported (4-bit).");
        }

        int groupSize = HfPackedBits.Int(settings, "group_size", HfPackedBits.Int(settings, "q_group_size", 128));
        var layouts = new List<HfIntegerLayout>();
        foreach (var (name, qweight) in tensors)
        {
            if (!name.EndsWith(".qweight", StringComparison.Ordinal) || qweight.DType != "I16" || qweight.Shape.Length != 2)
            {
                continue;
            }

            string stem = name[..^".qweight".Length];
            if (!tensors.TryGetValue(stem + ".qzeros", out var qzeros) || !tensors.TryGetValue(stem + ".scales", out var scales))
            {
                continue;
            }

            consumed.UnionWith([name, qzeros.Name, scales.Name]);
            long rows = qweight.Shape[0] * 4;
            long columns = qweight.Shape[1];
            if (columns % 64 != 0 || scales.Shape.Length != 2 || scales.Shape[1] != rows || !scales.Shape.SequenceEqual(qzeros.Shape))
            {
                throw new InvalidDataException($"{stem}: gemv_fast weight [{rows}, {columns}] does not match scales [{string.Join(", ", scales.Shape)}].");
            }

            long group = groupSize <= 0 ? columns : groupSize;
            long groups = (columns + group - 1) / group;
            nint words = (nint)qweight.Pointer;
            layouts.Add(new HfIntegerLayout
            {
                Name = stem + ".weight",
                Origin = qweight,
                Bits = 4,
                Rows = rows,
                Columns = columns,
                GroupSize = group,
                Affine = true,
                ScalesFitHalf = HfPackedBits.FitsHalf(scales) && HfPackedBits.FitsHalf(qzeros),
                DecodeRow = (n, rowCodes, rowScales, rowBiases) =>
                {
                    long n4 = n / 4;
                    long interleaved = n % 4 * 64;
                    ushort* row = (ushort*)words + n4 * columns;
                    for (long block = 0; block < columns / 64; block++)
                    {
                        for (int position = 0; position < 64; position++)
                        {
                            int at = (int)interleaved + position;
                            int code = (row[block * 64 + at / 4] >> (at % 4 * 4)) & 0xF;
                            rowCodes[(int)(block * 64 + position / 32 * 32 + s_column[position % 32])] = (byte)code;
                        }
                    }

                    for (long g = 0; g < groups; g++)
                    {
                        rowScales[(int)g] = HfPackedBits.Float(scales, g * rows + n);
                        rowBiases[(int)g] = HfPackedBits.Float(qzeros, g * rows + n);
                    }
                },
            });
        }

        return layouts;
    }

    /// <summary>
    /// Applies AutoAWQ's two column permutations of each 32-block (a 4 x 4 x 2 transpose, then the [0, 2, 4, 6, 1, 3, 5, 7] reorder of
    /// every eight) to the identity: entry p is the original column of the code stored at position p.
    /// </summary>
    private static int[] BuildColumnOrder()
    {
        var first = new int[32];
        for (int a = 0; a < 4; a++)
        {
            for (int b = 0; b < 4; b++)
            {
                for (int c = 0; c < 2; c++)
                {
                    first[b * 8 + a * 2 + c] = a * 8 + b * 2 + c;
                }
            }
        }

        var second = new int[32];
        for (int group = 0; group < 4; group++)
        {
            for (int x = 0; x < 4; x++)
            {
                for (int y = 0; y < 2; y++)
                {
                    second[group * 8 + y * 4 + x] = first[group * 8 + x * 2 + y];
                }
            }
        }

        return second;
    }
}
