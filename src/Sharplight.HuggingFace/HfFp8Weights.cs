using System.Text.Json;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.HuggingFace;

/// <summary>
/// FP8 checkpoints as vLLM loads them: E4M3 weights <c>X.weight</c> with a separate scale tensor are turned into self-contained
/// block-scaled weights before the architecture mapping sees them, so every mapping, fused-projection split and expert stack works on
/// them unchanged.
/// </summary>
/// <remarks>
/// <list type="bullet">
/// <item>Block-quantized (DeepSeek-V3 / Qwen FP8, <c>quantization_config.quant_method = "fp8"</c>, <c>weight_block_size [128, 128]</c>):
/// <c>X.weight_scale_inv</c> [ceil(N/128), ceil(K/128)]; the dequantized weight is <c>weight * weight_scale_inv</c> (a multiplier,
/// despite the name).</item>
/// <item>compressed-tensors and NVIDIA ModelOpt (<c>hf_quant_config.json</c>): <c>X.weight_scale</c> per tensor, per channel or per
/// block, also a multiplier. <c>X.input_scale</c> (static activation scales) is dropped: activations are quantized dynamically (or
/// not at all), so the per-tensor activation calibration is not needed. KV-cache scales (<c>k_scale</c> / <c>v_scale</c>) are dropped
/// too: FP8 KV caches are not used.</item>
/// </list>
/// A weight becomes <see cref="DataType.F8E4M3_B128"/> when K and the scale granularity along K are multiples of 128, else
/// <see cref="DataType.F8E4M3_B32"/> when they are multiples of 32 (the scale repeated in each block it covers), else
/// <see cref="DataType.BF16"/>. E4M3 / E5M2 tensors without a scale keep their codes as <see cref="DataType.F8E4M3"/> /
/// <see cref="DataType.F8E5M2"/>. Modules a checkpoint leaves unquantized are stored in BF16 and pass through.
/// </remarks>
internal static class HfFp8Weights
{
    private const string E4m3DType = "F8_E4M3";

    /// <summary>Scale tensor suffixes that replace <c>.weight</c> (DeepSeek style first).</summary>
    private static readonly string[] s_weightScaleSuffixes = [".weight_scale_inv", ".weight_scale"];

    /// <summary>Calibration tensors of FP8 checkpoints that the runtime does not use (activation and KV-cache scales).</summary>
    private static readonly string[] s_ignoredSuffixes = [".input_scale", ".k_scale", ".v_scale", ".q_scale", ".prob_scale"];

    /// <summary>
    /// The checkpoint tensors with FP8 weights converted and their scale tensors removed; <paramref name="tensors"/> itself when the
    /// checkpoint has no scaled FP8 weights.
    /// </summary>
    /// <exception cref="NotSupportedException">A scale tensor's shape fits no known scale layout.</exception>
    public static IReadOnlyDictionary<string, SafetensorsTensor> Resolve(HfConfig config, IReadOnlyDictionary<string, SafetensorsTensor> tensors)
    {
        var scaled = new Dictionary<string, SafetensorsTensor>(StringComparer.Ordinal);
        foreach (var (name, tensor) in tensors)
        {
            if (tensor.DType == E4m3DType && name.EndsWith(".weight", StringComparison.Ordinal) && FindScale(name, tensors) is { } scale)
            {
                scaled[name] = scale;
            }
        }

        if (scaled.Count == 0)
        {
            return tensors;
        }

        var block = DeclaredBlockSize(config);
        var consumed = scaled.Values.Select(s => s.Name).ToHashSet(StringComparer.Ordinal);
        var result = new Dictionary<string, SafetensorsTensor>(StringComparer.Ordinal);
        foreach (var (name, tensor) in tensors)
        {
            if (consumed.Contains(name) || s_ignoredSuffixes.Any(suffix => name.EndsWith(suffix, StringComparison.Ordinal)))
            {
                continue;
            }

            result[name] = scaled.TryGetValue(name, out var scale) ? Convert(tensor, scale, block) : tensor;
        }

        return result;
    }

    private static SafetensorsTensor? FindScale(string weightName, IReadOnlyDictionary<string, SafetensorsTensor> tensors)
    {
        string stem = weightName[..^".weight".Length];
        foreach (var suffix in s_weightScaleSuffixes)
        {
            if (tensors.TryGetValue(stem + suffix, out var scale))
            {
                return scale;
            }
        }

        return null;
    }

    /// <summary>
    /// The block size [rows, columns] the checkpoint declares: <c>quantization_config.weight_block_size</c> (fp8 method) or the
    /// <c>block_structure</c> of a compressed-tensors weight scheme; null when none is declared (it is then inferred from the shapes).
    /// </summary>
    private static (long Rows, long Columns)? DeclaredBlockSize(HfConfig config)
    {
        if (!config.Root.TryGetProperty("quantization_config", out var quantization) || quantization.ValueKind != JsonValueKind.Object)
        {
            return null;
        }

        if (Pair(quantization, "weight_block_size") is { } declared)
        {
            return declared;
        }

        if (quantization.TryGetProperty("config_groups", out var groups) && groups.ValueKind == JsonValueKind.Object)
        {
            foreach (var group in groups.EnumerateObject())
            {
                if (group.Value.ValueKind == JsonValueKind.Object && group.Value.TryGetProperty("weights", out var weights)
                    && weights.ValueKind == JsonValueKind.Object && Pair(weights, "block_structure") is { } structure)
                {
                    return structure;
                }
            }
        }

        return null;
    }

    private static (long, long)? Pair(JsonElement parent, string key)
        => parent.TryGetProperty(key, out var value) && value.ValueKind == JsonValueKind.Array && value.GetArrayLength() == 2
            ? (value[0].GetInt64(), value[1].GetInt64())
            : null;

    /// <summary>The FP8 weight <paramref name="weight"/> with its <paramref name="scale"/> folded into a block-scaled (or BF16) tensor.</summary>
    private static SafetensorsTensor Convert(SafetensorsTensor weight, SafetensorsTensor scale, (long Rows, long Columns)? block)
    {
        if (weight.Shape.Length < 2)
        {
            throw new NotSupportedException($"{weight.Name}: FP8 weight of rank {weight.Shape.Length}; scaled FP8 weights must be matrices.");
        }

        long rows = weight.Shape[^2];
        long columns = weight.Shape[^1];
        long batch = weight.Shape[..^2].Aggregate(1L, (a, d) => a * d);
        var layout = HfFp8ScaleLayout.Infer(batch, rows, columns, scale.Shape, block)
            ?? throw new NotSupportedException($"{weight.Name}: scale {scale.Name} of shape [{string.Join(", ", scale.Shape)}] fits no FP8 scale layout.");
        var target = TargetType(columns, layout.ColumnBlock);
        var codes = weight.ToWeightInfo();
        var converted = new WeightInfo
        {
            Name = weight.Name,
            Type = target,
            Shape = codes.Shape,
            Source = new HfFp8WeightSource(codes.Source, scale.ToWeightInfo(), layout, target, rows, columns),
        };
        return new SafetensorsTensor
        {
            Name = weight.Name,
            DType = weight.DType,
            Type = target,
            Shape = weight.Shape,
            Offset = weight.Offset,
            ByteLength = converted.ByteSize,
            File = weight.File,
            Converted = converted,
        };
    }

    /// <summary>The storage type of a weight with <paramref name="columns"/> values per row whose scales cover <paramref name="columnBlock"/> of them.</summary>
    private static DataType TargetType(long columns, long columnBlock)
    {
        bool oneScalePerRow = columnBlock >= columns;
        foreach (var type in new[] { DataType.F8E4M3_B128, DataType.F8E4M3_B32 })
        {
            if (columns % type.BlockSize == 0 && (oneScalePerRow || columnBlock % type.BlockSize == 0))
            {
                return type;
            }
        }

        return DataType.BF16;
    }
}
