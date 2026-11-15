namespace Sharplight.Import.HuggingFace;

/// <summary>
/// torchao checkpoints (<c>quantization_config.quant_method = "torchao"</c>, saved through transformers with safetensors, torchao 0.18): a
/// quantized weight <c>X.weight</c> is stored as the parts of its tensor subclass, <c>X._weight_qdata</c>, <c>X._weight_scale</c> and, for
/// integer weights, <c>X._weight_zero_point</c>:
/// <list type="bullet">
/// <item><c>Int8WeightOnlyConfig</c> / <c>Int8DynamicActivationInt8WeightConfig</c> (<c>Int8Tensor</c>): I8 [N, K] codes, scales [N, K / group size] (per row:
/// [N, 1]), I8 zero points of the same shape; value = (code - zero point) * scale. The codes become unsigned ones with the zero point + 128
/// (<see cref="HfIntegerWeights"/>); activations are not quantized here. torchao stores f32 scales: those of a BF16 model are BF16 values (f16-exact in practice) and keep the integer
/// block type, scales that are not f16 values decode to BF16.</item>
/// <item><c>Float8WeightOnlyConfig</c> / <c>Float8DynamicActivationFloat8WeightConfig</c> (<c>Float8Tensor</c>): F8_E4M3 codes with f32 scales,
/// renamed to <c>X.weight</c> / <c>X.weight_scale</c> for the FP8 path (<see cref="HfFp8Weights"/>).</item>
/// </list>
/// The int4 / intx tensors (<c>Int4Tensor</c>, <c>IntxUnpackedToInt8Tensor</c>), the NF4 tensor and the pre-0.9 <c>AffineQuantizedTensor</c> layouts
/// are not read.
/// </summary>
internal static unsafe class HfTorchAoLayouts
{
    private const string Codes = "._weight_qdata";

    /// <summary>The checkpoint tensors with Float8 weights renamed to the FP8 scale layout (the others unchanged).</summary>
    /// <param name="tensors">The checkpoint tensors.</param>
    public static IReadOnlyDictionary<string, SafetensorsTensor> RenameFloat8(IReadOnlyDictionary<string, SafetensorsTensor> tensors)
    {
        var result = new Dictionary<string, SafetensorsTensor>(StringComparer.Ordinal);
        bool any = false;
        foreach (var (name, tensor) in tensors)
        {
            string? stem = name.EndsWith(Codes, StringComparison.Ordinal) ? name[..^Codes.Length] : null;
            if (stem is not null && tensor.DType == "F8_E4M3" && tensors.ContainsKey(stem + "._weight_scale"))
            {
                result[stem + ".weight"] = HfQuantoLayouts.Renamed(tensor, stem + ".weight");
                any = true;
            }
            else if (name.EndsWith("._weight_scale", StringComparison.Ordinal) && tensors.TryGetValue(name[..^"._weight_scale".Length] + Codes, out var codes)
                && codes.DType == "F8_E4M3")
            {
                result[name[..^"._weight_scale".Length] + ".weight_scale"] = HfQuantoLayouts.Renamed(tensor, name[..^"._weight_scale".Length] + ".weight_scale");
            }
            else
            {
                result[name] = tensor;
            }
        }

        return any ? result : tensors;
    }

    /// <summary>The layouts of the int8 torchao modules among <paramref name="tensors"/>; their tensors are added to <paramref name="consumed"/>.</summary>
    /// <param name="tensors">The checkpoint tensors.</param>
    /// <param name="consumed">Receives the tensors the layouts replace.</param>
    /// <exception cref="NotSupportedException">The scale layout does not fit the weight.</exception>
    public static List<HfIntegerLayout> Find(IReadOnlyDictionary<string, SafetensorsTensor> tensors, HashSet<string> consumed)
    {
        var layouts = new List<HfIntegerLayout>();
        foreach (var (name, data) in tensors)
        {
            if (!name.EndsWith(Codes, StringComparison.Ordinal) || data.DType != "I8" || data.Shape.Length != 2)
            {
                continue;
            }

            string stem = name[..^Codes.Length];
            if (!tensors.TryGetValue(stem + "._weight_scale", out var scale))
            {
                continue;
            }

            tensors.TryGetValue(stem + "._weight_zero_point", out var zero);
            consumed.UnionWith([name, scale.Name, .. zero is null ? Array.Empty<string>() : [zero.Name]]);
            long rows = data.Shape[0];
            long columns = data.Shape[1];
            long scaleCount = scale.Shape.Aggregate(1L, (a, d) => a * d);
            long groups = scaleCount / rows;
            if (scaleCount % rows != 0 || groups == 0 || columns % groups != 0 || (zero is not null && zero.Shape.Aggregate(1L, (a, d) => a * d) != scaleCount))
            {
                throw new NotSupportedException($"{stem}: torchao scales [{string.Join(", ", scale.Shape)}] do not fit the weight [{rows}, {columns}].");
            }

            nint codes = (nint)data.Pointer;
            nint zeros = zero is null ? 0 : (nint)zero.Pointer;
            layouts.Add(new HfIntegerLayout
            {
                Name = stem + ".weight",
                Origin = data,
                Bits = 8,
                Rows = rows,
                Columns = columns,
                GroupSize = columns / groups,
                ScalesFitHalf = HfPackedBits.FitsHalf(scale),
                DecodeRow = (n, rowCodes, rowScales, rowZeros) =>
                {
                    byte* row = (byte*)codes + n * columns;
                    for (long k = 0; k < columns; k++)
                    {
                        rowCodes[(int)k] = (byte)(row[k] ^ 0x80);   // the signed code plus 128
                    }

                    for (long g = 0; g < groups; g++)
                    {
                        rowScales[(int)g] = HfPackedBits.Float(scale, n * groups + g);
                        rowZeros[(int)g] = 128 + (zeros == 0 ? 0 : *((sbyte*)zeros + n * groups + g));
                    }
                },
            });
        }

        return layouts;
    }
}
