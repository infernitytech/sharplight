using System.Text.Json;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// EETQ checkpoints (<c>quantization_config.quant_method = "eetq"</c>, int8 weight-only): per linear <c>X.weight</c> I8 [K, N]
/// (input features first) and <c>X.weight_scales</c> F16 [N], one scale per output channel, value = q * scale (the scale is
/// the column's absolute maximum divided by 128). The int8 matrix is not row-major: EETQ's quantizer
/// (csrc/cutlass_kernels/cutlass_preprocessors.cc, FasterTransformer's weight preprocessing for the Ampere mixed-type GEMM) permutes
/// the rows of every 16-row tile, transposes the matrix, interleaves pairs of 64-row column tiles, adds 128 to every byte and swaps
/// the middle two bytes of every four. This reader inverts that closed-form (checked against the library's own code for every position).
/// The weights become the integer block types with the zero point 128 (<see cref="HfIntegerWeights.TargetType"/>): code = q + 128,
/// value = (code - 128) * scale, exact in f32 as in the library's f16 product.
/// </summary>
internal static unsafe class HfEetqLayouts
{
    /// <summary>The 16-row tile permutation of the preprocessing: stored tile row of each logical row.</summary>
    private static readonly int[] s_tileRow = BuildTileRows();

    private static int[] BuildTileRows()
    {
        var inverse = new int[16];
        for (int t = 0; t < 16; t++)
        {
            inverse[8 * (t % 4 / 2) + t % 2 + 2 * (t / 4)] = t;
        }

        return inverse;
    }

    /// <summary>The layouts of the EETQ modules among <paramref name="tensors"/>; their tensors are added to <paramref name="consumed"/>.</summary>
    /// <exception cref="NotSupportedException">The shape does not meet EETQ's tiling (K and N multiples of 64).</exception>
    public static List<HfIntegerLayout> Find(JsonElement settings, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HashSet<string> consumed)
    {
        string? weights = HfPackedBits.String(settings, "weights");
        if (weights is not (null or "int8"))
        {
            throw new NotSupportedException($"EETQ weights '{weights}' are not supported (int8).");
        }

        var layouts = new List<HfIntegerLayout>();
        foreach (var (name, scales) in tensors)
        {
            if (!name.EndsWith(".weight_scales", StringComparison.Ordinal) || !tensors.TryGetValue(name[..^"_scales".Length], out var codes)
                || codes.DType != "I8" || codes.Shape.Length != 2 || scales.Shape.Length != 1 || scales.Shape[0] != codes.Shape[1])
            {
                continue;
            }

            long columns = codes.Shape[0];
            long rows = codes.Shape[1];
            if (columns % 64 != 0 || rows % 64 != 0)
            {
                throw new NotSupportedException($"{codes.Name}: EETQ weights are tiled in 64s; found [{columns}, {rows}].");
            }

            consumed.UnionWith([name, codes.Name]);
            nint raw = (nint)codes.Pointer;
            long vectorRows = columns / 4;
            layouts.Add(new HfIntegerLayout
            {
                Name = codes.Name,
                Origin = codes,
                Bits = 8,
                Rows = rows,
                Columns = columns,
                GroupSize = columns,
                ScalesFitHalf = HfPackedBits.FitsHalf(scales),
                DecodeRow = (n, rowCodes, rowScales, rowZeros) =>
                {
                    byte* data = (byte*)raw;
                    for (long k = 0; k < columns; k++)
                    {
                        long stored = k / 16 * 16 + s_tileRow[k % 16];
                        long vr = stored / 4;
                        long word = n / 2 * 2 * vectorRows + 2 * (16 * (vr / 16)) + 16 * (n % 2) + vr % 16;
                        int inWord = (int)(stored % 4) switch { 1 => 2, 2 => 1, var j => j };
                        // The stored byte is q + 128 (the preprocessing's bias), the unsigned code of the zero point 128.
                        rowCodes[(int)k] = data[4 * word + inWord];
                    }

                    rowScales[0] = HfPackedBits.Float(scales, n);
                    rowZeros[0] = 128;
                },
            });
        }

        return layouts;
    }
}
