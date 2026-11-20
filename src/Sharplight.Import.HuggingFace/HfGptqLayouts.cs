using System.Text.Json;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// GPTQ checkpoints (auto-gptq, gptqmodel; <c>quantization_config.quant_method = "gptq"</c>): per module <c>X.qweight</c> I32
/// [K * bits / 32, N] (codes packed along K, a little-endian bit stream per column, so 3-bit codes straddle words),
/// <c>X.qzeros</c> I32 [groups, N * bits / 32] (packed along N), <c>X.scales</c> F16 [groups, N] and optionally <c>X.g_idx</c> I32
/// [K] (the group of each column; act-order when it is not k / group_size). value = (code - zero) * scale. The original GPTQ
/// format (<c>checkpoint_format</c> <c>gptq</c> or absent) stores every zero point minus one, so the stored value plus one is the
/// zero point (a zero point of 0 thus reads as 2^bits, as auto-gptq and gptqmodel decode it); <c>gptq_v2</c> stores them as they are.
/// A symmetric checkpoint (<c>sym: true</c>) has the zero point 2^(bits-1) whatever <c>qzeros</c> holds, as vLLM's GPTQ-Marlin kernels
/// read it: Qwen's official GPTQ-Int4 checkpoints (Qwen3.5-122B-A10B) store 8 in the original format, which would read as 9.
/// </summary>
internal static unsafe class HfGptqLayouts
{
    /// <summary>The layouts of the GPTQ modules among <paramref name="tensors"/>; their tensors are added to <paramref name="consumed"/>.</summary>
    /// <exception cref="NotSupportedException">The bit width is not 2, 3, 4 or 8.</exception>
    public static List<HfIntegerLayout> Find(JsonElement settings, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HashSet<string> consumed)
    {
        int bits = HfPackedBits.Int(settings, "bits", 4);
        if (bits is not (2 or 3 or 4 or 8))
        {
            throw new NotSupportedException($"GPTQ with {bits}-bit codes is not supported (2, 3, 4 or 8).");
        }

        int groupSize = HfPackedBits.Int(settings, "group_size", 128);
        string format = HfPackedBits.String(settings, "checkpoint_format") ?? "gptq";
        if (format is not ("gptq" or "gptq_v2"))
        {
            throw new NotSupportedException($"GPTQ checkpoint format '{format}' is not supported (gptq, gptq_v2).");
        }

        int zeroOffset = format == "gptq" ? 1 : 0;
        int? symmetricZero = settings.ValueKind == JsonValueKind.Object && settings.TryGetProperty("sym", out var sym) && sym.ValueKind == JsonValueKind.True
            ? 1 << (bits - 1)
            : null;
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

            long rows = qweight.Shape[1];
            long columns = qweight.Shape[0] * 32 / bits;
            tensors.TryGetValue(stem + ".g_idx", out var gIdx);
            consumed.UnionWith([name, qzeros.Name, scales.Name, .. gIdx is null ? Array.Empty<string>() : [gIdx.Name]]);
            nint codes = (nint)qweight.Pointer;
            nint zeroWords = (nint)qzeros.Pointer;
            long zeroRowWords = qzeros.Shape[1];
            long groups = scales.Shape[0];
            layouts.Add(new HfIntegerLayout
            {
                Name = stem + ".weight",
                Origin = qweight,
                Bits = bits,
                Rows = rows,
                Columns = columns,
                GroupSize = groupSize <= 0 ? columns : groupSize,
                GroupIndex = gIdx is null ? null : HfPackedBits.Int32s(gIdx),
                ScalesFitHalf = HfPackedBits.FitsHalf(scales),
                DecodeRow = (n, rowCodes, rowScales, rowZeros) =>
                {
                    for (long k = 0; k < columns; k++)
                    {
                        rowCodes[(int)k] = (byte)HfPackedBits.Code((byte*)codes + n * 4, rows, k * bits, bits);
                    }

                    for (long g = 0; g < groups; g++)
                    {
                        rowScales[(int)g] = HfPackedBits.Float(scales, g * rows + n);
                        rowZeros[(int)g] = symmetricZero ?? HfPackedBits.Code((byte*)zeroWords + g * zeroRowWords * 4, 1, n * bits, bits) + zeroOffset;
                    }
                },
            });
        }

        return layouts;
    }
}
