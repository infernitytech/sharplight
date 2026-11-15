using System.Text.Json;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// AMD Quark integer weights (<c>int4</c> / <c>uint4</c> / <c>int8</c> / <c>uint8</c>; the spec of the module in the config says which),
/// value = (q - zero point) * scale with q and the zero point signed for the <c>int</c> types:
/// <list type="bullet">
/// <item><c>per_group</c>: Quark transposes the weight: <c>X.weight</c> [K, N / 8] I32 for 4-bit codes (eight codes of consecutive output
/// rows per word, nibble i holding row 8w + [0, 2, 4, 6, 1, 3, 5, 7][i] when <c>export.pack_method</c> is <c>reorder</c>, else row 8w + i,
/// the AWQ GEMM packing) or [K, N] I8 / U8 for 8-bit codes; <c>X.weight_scale</c> [K / group, N]; <c>X.weight_zero_point</c> [K / group,
/// N / 8] I32 (4-bit, same packing) or [K / group, N] (8-bit), absent for symmetric weights.</item>
/// <item><c>per_channel</c>: <c>X.weight</c> [N, K / 8] I32 (codes packed along K the same way) or [N, K]; <c>X.weight_scale</c> [N];
/// <c>X.weight_zero_point</c> [N / 8] I32 or [N].</item>
/// <item><c>per_tensor</c>: as <c>per_channel</c> with one scalar scale (and zero point).</item>
/// </list>
/// Signed codes are re-based to unsigned ones (adding 8 or 128, to the zero point as well), so the weight keeps its codes in the integer
/// block types (<see cref="HfIntegerWeights"/>); scales that are not exact f16 values (Quark writes f32) decode to BF16.
/// </summary>
internal static unsafe class HfQuarkLayouts
{
    /// <summary>The nibble that holds column c % 8 of a reordered word (the inverse of [0, 2, 4, 6, 1, 3, 5, 7]).</summary>
    private static readonly int[] s_reorderedNibble = [0, 4, 1, 5, 2, 6, 3, 7];

    private static readonly int[] s_plainNibble = [0, 1, 2, 3, 4, 5, 6, 7];

    /// <summary>The layouts of the integer Quark modules among <paramref name="tensors"/>; their tensors are added to <paramref name="consumed"/>.</summary>
    /// <exception cref="InvalidDataException">A module's tensors do not match its spec.</exception>
    public static List<HfIntegerLayout> Find(JsonElement settings, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HashSet<string> consumed)
    {
        int[] nibble = HfQuarkSpecs.Reorder(settings) ? s_reorderedNibble : s_plainNibble;
        var layouts = new List<HfIntegerLayout>();
        foreach (var (name, weight) in tensors)
        {
            if (!name.EndsWith(".weight", StringComparison.Ordinal) || weight.Shape.Length != 2 || weight.DType is not ("I32" or "I8" or "U8"))
            {
                continue;
            }

            string stem = name[..^".weight".Length];
            if (!tensors.TryGetValue(stem + ".weight_scale", out var scale) || scale.DType is not ("F16" or "BF16" or "F32")
                || HfQuarkSpecs.WeightSpec(settings, stem) is not { } spec)
            {
                continue;
            }

            string dtype = HfPackedBits.String(spec, "dtype") ?? "";
            if (dtype is not ("int4" or "uint4" or "int8" or "uint8"))
            {
                continue;
            }

            bool signed = dtype.StartsWith("int", StringComparison.Ordinal);
            int bits = dtype.EndsWith('4') ? 4 : 8;
            if ((bits == 4) != (weight.DType == "I32"))
            {
                continue;
            }

            string scheme = HfPackedBits.String(spec, "qscheme") ?? "per_tensor";
            tensors.TryGetValue(stem + ".weight_zero_point", out var zeroPoint);
            consumed.UnionWith([name, scale.Name, .. zeroPoint is null ? Array.Empty<string>() : [zeroPoint.Name]]);
            layouts.Add(Build(stem, weight, scale, zeroPoint, scheme, bits, signed, nibble));
        }

        return layouts;
    }

    private static HfIntegerLayout Build(string stem, SafetensorsTensor weight, SafetensorsTensor scale, SafetensorsTensor? zeroPoint,
        string scheme, int bits, bool signed, int[] nibble)
    {
        bool grouped = scheme == "per_group";
        long rows = grouped ? (bits == 4 ? weight.Shape[1] * 8 : weight.Shape[1]) : weight.Shape[0];
        long columns = grouped ? weight.Shape[0] : bits == 4 ? weight.Shape[1] * 8 : weight.Shape[1];
        long scaleCount = scale.Shape.Aggregate(1L, (a, d) => a * d);
        long groups = grouped ? scale.Shape[0] : 1;
        long expected = grouped ? groups * rows : scheme == "per_channel" ? rows : 1;
        if (scaleCount != expected || columns % groups != 0)
        {
            throw new InvalidDataException($"{stem}: Quark {scheme} scales [{string.Join(", ", scale.Shape)}] do not match the weight [{rows}, {columns}].");
        }

        long group = columns / groups;
        int signBias = signed ? 1 << (bits - 1) : 0;
        nint codes = (nint)weight.Pointer;
        nint zeros = zeroPoint is null ? 0 : (nint)zeroPoint.Pointer;
        long weightRowWords = grouped ? rows / 8 : columns / 8;

        // The raw (unsigned bit pattern) code at (row, column) and zero point at (row, group).
        int Code(long n, long k)
        {
            if (bits == 8)
            {
                return *((byte*)codes + (grouped ? k * rows + n : n * columns + k));
            }

            return grouped
                ? (int)((*((uint*)codes + k * weightRowWords + n / 8) >> (4 * nibble[n % 8])) & 0xF)
                : (int)((*((uint*)codes + n * weightRowWords + k / 8) >> (4 * nibble[k % 8])) & 0xF);
        }

        int Zero(long n, long g)
        {
            if (zeros == 0)
            {
                return 0;
            }

            if (bits == 8)
            {
                return *((byte*)zeros + (grouped ? g * rows + n : scheme == "per_channel" ? n : 0));
            }

            return grouped ? (int)((*((uint*)zeros + g * (rows / 8) + n / 8) >> (4 * nibble[n % 8])) & 0xF)
                : scheme == "per_channel" ? (int)((*((uint*)zeros + n / 8) >> (4 * nibble[n % 8])) & 0xF)
                : (int)(*(uint*)zeros & 0xF);
        }

        // Signed patterns -> value + 2^(bits - 1): flipping the sign bit (xor) does exactly that.
        int flip = signBias;
        return new HfIntegerLayout
        {
            Name = stem + ".weight",
            Origin = weight,
            Bits = bits,
            Rows = rows,
            Columns = columns,
            GroupSize = group,
            ScalesFitHalf = HfPackedBits.FitsHalf(scale),
            DecodeRow = (n, rowCodes, rowScales, rowZeros) =>
            {
                for (long k = 0; k < columns; k++)
                {
                    rowCodes[(int)k] = (byte)(Code(n, k) ^ flip);
                }

                for (long g = 0; g < groups; g++)
                {
                    rowScales[(int)g] = HfPackedBits.Float(scale, grouped ? g * rows + n : scheme == "per_channel" ? n : 0);
                    rowZeros[(int)g] = Zero(n, g) ^ flip;
                }
            },
        };
    }
}
