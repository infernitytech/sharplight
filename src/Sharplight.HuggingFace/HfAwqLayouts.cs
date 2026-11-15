using System.Text.Json;

namespace Sharplight.HuggingFace;

/// <summary>
/// AWQ checkpoints (AutoAWQ; <c>quantization_config.quant_method = "awq"</c>, 4-bit codes, value = (code - zero) * scale):
/// <list type="bullet">
/// <item><c>gemm</c> (the default): <c>X.qweight</c> I32 [K, N / 8] and <c>X.qzeros</c> I32 [groups, N / 8], codes packed along N
/// in the interleaved order 0, 2, 4, 6, 1, 3, 5, 7 (nibble i of a word holds column 8w + order[i]); <c>X.scales</c> F16 [groups, N].</item>
/// <item><c>gemv</c>: <c>X.qweight</c> I32 [N, K / 8] packed along K in order, <c>X.qzeros</c> I32 [N, W] (group g in nibble g % 8
/// of word g / 8), <c>X.scales</c> F16 [N, 8 W] (padded past the last group).</item>
/// </list>
/// <c>gemv_fast</c>, <c>exllama</c> and <c>marlin</c> repacks are not read.
/// </summary>
internal static unsafe class HfAwqLayouts
{
    /// <summary>The nibble that holds column c % 8 of a GEMM-packed word (AutoAWQ's AWQ_REVERSE_ORDER).</summary>
    private static readonly int[] s_reverseOrder = [0, 4, 1, 5, 2, 6, 3, 7];

    /// <summary>The layouts of the AWQ modules among <paramref name="tensors"/>; their tensors are added to <paramref name="consumed"/>.</summary>
    /// <exception cref="NotSupportedException">The packing version or bit width is not read.</exception>
    public static List<HfIntegerLayout> Find(JsonElement settings, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HashSet<string> consumed)
    {
        int bits = HfPackedBits.Int(settings, "bits", HfPackedBits.Int(settings, "w_bit", 4));
        string version = (HfPackedBits.String(settings, "version") ?? "gemm").ToLowerInvariant();
        if (bits != 4 || version is not ("gemm" or "gemv"))
        {
            throw new NotSupportedException($"AWQ {version} with {bits}-bit codes is not supported (gemm and gemv, 4-bit).");
        }

        int groupSize = HfPackedBits.Int(settings, "group_size", HfPackedBits.Int(settings, "q_group_size", 128));
        var layouts = new List<HfIntegerLayout>();
        foreach (var (name, qweight) in tensors)
        {
            if (!name.EndsWith(".qweight", StringComparison.Ordinal) || qweight.DType != "I32" || qweight.Shape.Length != 2)
            {
                continue;
            }

            string stem = name[..^".qweight".Length];
            if (!tensors.TryGetValue(stem + ".qzeros", out var qzeros) || !tensors.TryGetValue(stem + ".scales", out var scales))
            {
                continue;
            }

            consumed.UnionWith([name, qzeros.Name, scales.Name]);
            bool gemm = version == "gemm";
            long rows = gemm ? qweight.Shape[1] * 8 : qweight.Shape[0];
            long columns = gemm ? qweight.Shape[0] : qweight.Shape[1] * 8;
            long group = groupSize <= 0 ? columns : groupSize;
            long groups = (columns + group - 1) / group;
            nint codes = (nint)qweight.Pointer;
            nint zeroWords = (nint)qzeros.Pointer;
            long zeroRowWords = qzeros.Shape[1];
            long scaleRow = scales.Shape[1];
            layouts.Add(new HfIntegerLayout
            {
                Name = stem + ".weight",
                Origin = qweight,
                Bits = 4,
                Rows = rows,
                Columns = columns,
                GroupSize = group,
                ScalesFitHalf = HfPackedBits.FitsHalf(scales),
                DecodeRow = gemm
                    ? (n, rowCodes, rowScales, rowZeros) =>
                    {
                        int shift = 4 * s_reverseOrder[n % 8];
                        long word = n / 8;
                        for (long k = 0; k < columns; k++)
                        {
                            rowCodes[(int)k] = (byte)((*((uint*)codes + k * (rows / 8) + word) >> shift) & 0xF);
                        }

                        for (long g = 0; g < groups; g++)
                        {
                            rowScales[(int)g] = HfPackedBits.Float(scales, g * rows + n);
                            rowZeros[(int)g] = (*((uint*)zeroWords + g * zeroRowWords + word) >> shift) & 0xF;
                        }
                    }
                    : (n, rowCodes, rowScales, rowZeros) =>
                    {
                        for (long k = 0; k < columns; k++)
                        {
                            rowCodes[(int)k] = (byte)HfPackedBits.Code((byte*)codes + n * (columns / 8) * 4, 1, k * 4, 4);
                        }

                        for (long g = 0; g < groups; g++)
                        {
                            rowScales[(int)g] = HfPackedBits.Float(scales, n * scaleRow + g);
                            rowZeros[(int)g] = HfPackedBits.Code((byte*)zeroWords + n * zeroRowWords * 4, 1, g * 4, 4);
                        }
                    },
            });
        }

        return layouts;
    }
}
