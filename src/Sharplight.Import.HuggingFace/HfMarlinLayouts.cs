using System.Text.Json;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Marlin-format checkpoints, GPTQ (<c>checkpoint_format</c> <c>marlin</c>, GPTQModel / vLLM exports) and AWQ (<c>version</c>
/// <c>marlin</c>, AutoAWQ): symmetric 4 or 8-bit codes (zero point 2^(bits - 1)) repacked for the Marlin kernel. Per module
/// <c>X.qweight</c> (the original IST-DASLab Marlin checkpoints call it <c>X.B</c>) I32 [K / 16, N * 16 * bits / 32] and <c>X.scales</c>
/// (<c>X.s</c>) F16 [K / group, N]; value = (code - 2^(bits - 1)) * scale. The importer undoes the repack:
/// <list type="bullet">
/// <item>codes: the [K, N] matrix is cut into 16 x 16 tiles, each tile row of the matrix becomes one row of N * 16 values (tile by tile,
/// row-major inside a tile), every 1024 values are shuffled by Marlin's tensor-core permutation (then the [0, 2, 4, 6, 1, 3, 5, 7] interleave of
/// every eight for 4-bit, [0, 2, 1, 3] of every four for 8-bit) and value i of each word of 32 bits goes to bits i * bits;</item>
/// <item>scales: every 64 values (every 32 when there is one group per column) of the group row are shuffled by Marlin's scale permutation.</item>
/// </list>
/// Act-order (a non-trivial <c>g_idx</c>) is not read. The workspace tensor and the trivial <c>g_idx</c> are dropped.
/// </summary>
internal static unsafe class HfMarlinLayouts
{
    private const int Tile = 16;

    /// <summary>The checkpoint formats of a GPTQ configuration that name Marlin.</summary>
    /// <param name="settings">The <c>quantization_config</c>.</param>
    public static bool IsMarlin(JsonElement settings)
        => (HfPackedBits.String(settings, "checkpoint_format") ?? HfPackedBits.String(settings, "format"))?.ToLowerInvariant() is "marlin" or "gptq_marlin";

    /// <summary>For each position of a 1024-value chunk of a tile row: where the value is stored, for 4-bit and 8-bit codes.</summary>
    private static readonly int[] s_inverse4 = BuildInverse(4);

    private static readonly int[] s_inverse8 = BuildInverse(8);

    /// <summary>The Marlin scale permutation of 64 values (several groups) and of 32 values (one group per column), inverted.</summary>
    private static readonly int[] s_scaleInverse = BuildScaleInverse(64);

    private static readonly int[] s_scaleSingleInverse = BuildScaleInverse(32);

    /// <summary>The layouts of the Marlin modules among <paramref name="tensors"/>; their tensors are added to <paramref name="consumed"/>.</summary>
    /// <exception cref="NotSupportedException">The bit width is not 4 or 8, or a module uses act-order.</exception>
    public static List<HfIntegerLayout> Find(JsonElement settings, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HashSet<string> consumed)
    {
        int bits = HfPackedBits.Int(settings, "bits", HfPackedBits.Int(settings, "w_bit", 4));
        if (bits is not (4 or 8))
        {
            throw new NotSupportedException($"Marlin with {bits}-bit codes is not supported (4 or 8).");
        }

        int groupSize = HfPackedBits.Int(settings, "group_size", HfPackedBits.Int(settings, "q_group_size", 128));
        int[] inverse = bits == 4 ? s_inverse4 : s_inverse8;
        int pack = 32 / bits;
        int zero = 1 << (bits - 1);
        var layouts = new List<HfIntegerLayout>();
        foreach (var (name, qweight) in tensors)
        {
            bool original = name.EndsWith(".B", StringComparison.Ordinal);
            if (!(name.EndsWith(".qweight", StringComparison.Ordinal) || original) || qweight.DType != "I32" || qweight.Shape.Length != 2)
            {
                continue;
            }

            string stem = name[..name.LastIndexOf('.')];
            if (!tensors.TryGetValue(stem + (original ? ".s" : ".scales"), out var scales) || tensors.ContainsKey(stem + ".qzeros"))
            {
                continue;
            }

            long columns = qweight.Shape[0] * Tile;
            long rows = qweight.Shape[1] * pack / Tile;
            if (scales.Shape.Length != 2 || scales.Shape[1] != rows || rows % Tile != 0 || columns % Tile != 0)
            {
                throw new InvalidDataException($"{stem}: Marlin weight [{columns}, {rows}] does not match scales [{string.Join(", ", scales.Shape)}].");
            }

            consumed.UnionWith([name, scales.Name]);
            if (tensors.TryGetValue(stem + ".g_idx", out var gIdx))
            {
                int[] index = HfPackedBits.Int32s(gIdx);
                long step = groupSize <= 0 ? columns : groupSize;
                if (Enumerable.Range(0, index.Length).Any(k => index[k] != k / step))
                {
                    throw new NotSupportedException($"{stem}: Marlin checkpoints with act-order (g_idx) are not supported.");
                }

                consumed.Add(gIdx.Name);
            }

            foreach (string extra in (string[])[stem + ".workspace", stem + ".g_idx_sort_indices"])
            {
                if (tensors.ContainsKey(extra))
                {
                    consumed.Add(extra);
                }
            }

            long group = groupSize <= 0 || groupSize >= columns ? columns : groupSize;
            long groups = columns / group;
            int[] scaleInverse = group < columns ? s_scaleInverse : s_scaleSingleInverse;
            if (scales.Shape[0] != groups)
            {
                throw new InvalidDataException($"{stem}: Marlin scales have {scales.Shape[0]} groups, expected {groups}.");
            }

            nint words = (nint)qweight.Pointer;
            long rowWords = rows * Tile / pack;
            layouts.Add(new HfIntegerLayout
            {
                Name = stem + ".weight",
                Origin = qweight,
                Bits = bits,
                Rows = rows,
                Columns = columns,
                GroupSize = group,
                ScalesFitHalf = HfPackedBits.FitsHalf(scales),
                DecodeRow = (n, rowCodes, rowScales, rowZeros) =>
                {
                    long tileColumn = n / Tile * (Tile * Tile) + n % Tile;
                    for (long k = 0; k < columns; k++)
                    {
                        long position = tileColumn + k % Tile * Tile;
                        long stored = position / 1024 * 1024 + inverse[position % 1024];
                        uint word = *((uint*)words + k / Tile * rowWords + stored / pack);
                        rowCodes[(int)k] = (byte)((word >> (int)(stored % pack * bits)) & ((1u << bits) - 1));
                    }

                    int period = scaleInverse.Length;
                    long source = n / period * period + scaleInverse[n % period];
                    for (long g = 0; g < groups; g++)
                    {
                        rowScales[(int)g] = HfPackedBits.Float(scales, g * rows + source);
                        rowZeros[(int)g] = zero;
                    }
                },
            });
        }

        return layouts;
    }

    /// <summary>Marlin's tensor-core permutation of 1024 values (vLLM <c>get_perms</c>) with the bit-width interleave, inverted.</summary>
    private static int[] BuildInverse(int bits)
    {
        var permutation = new List<int>();
        for (int i = 0; i < 32; i++)
        {
            var first = new List<int>();
            int col = i / 4;
            for (int block = 0; block < 2; block++)
            {
                foreach (int row in (int[])[2 * (i % 4), 2 * (i % 4) + 1, 2 * (i % 4 + 4), 2 * (i % 4 + 4) + 1])
                {
                    first.Add(16 * row + col + 8 * block);
                }
            }

            for (int j = 0; j < 4; j++)
            {
                permutation.AddRange(first.Select(p => p + 256 * j));
            }
        }

        int[] interleave = bits == 4 ? [0, 2, 4, 6, 1, 3, 5, 7] : [0, 2, 1, 3];
        var stored = new int[permutation.Count];
        for (int j = 0; j < stored.Length; j++)
        {
            // res[j] = in[perm[j]] with perm = permutation reshaped by rows of interleave.Length and column-selected by interleave.
            int chunk = j / interleave.Length * interleave.Length;
            stored[j] = permutation[chunk + interleave[j % interleave.Length]];
        }

        var inverse = new int[stored.Length];
        for (int j = 0; j < stored.Length; j++)
        {
            inverse[stored[j]] = j;
        }

        return inverse;
    }

    /// <summary>Marlin's scale permutation (<c>scale_perm</c> for 64 values, <c>scale_perm_single</c> for 32), inverted.</summary>
    private static int[] BuildScaleInverse(int period)
    {
        var permutation = new List<int>();
        if (period == 64)
        {
            for (int i = 0; i < 8; i++)
            {
                for (int j = 0; j < 8; j++)
                {
                    permutation.Add(i + 8 * j);
                }
            }
        }
        else
        {
            for (int i = 0; i < 4; i++)
            {
                foreach (int j in (int[])[0, 1, 8, 9, 16, 17, 24, 25])
                {
                    permutation.Add(2 * i + j);
                }
            }
        }

        var inverse = new int[period];
        for (int j = 0; j < period; j++)
        {
            inverse[permutation[j]] = j;
        }

        return inverse;
    }
}
