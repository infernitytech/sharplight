using System.Text.Json;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.HuggingFace;

/// <summary>
/// compressed-tensors MXFP4 checkpoints (<c>quantization_config.format = "mxfp4-pack-quantized"</c>, e.g. Kimi K3's routed experts):
/// each quantized linear stores <c>X.weight_packed</c> (U8 [out, in/2], element 2i in the low nibble of byte i, E2M1 codes) and
/// <c>X.weight_scale</c> (U8 [out, in/32], E8M0 exponents biased by 127, one per 32 values). Before the architecture mapping sees them,
/// every such pair becomes one self-contained <see cref="DataType.MXFP4"/> weight <c>X.weight</c> [out, in], so mappings and expert stacks
/// work on it unchanged. Other tensors pass through.
/// </summary>
internal static class HfMxfp4Weights
{
    private const string PackedSuffix = ".weight_packed";
    private const string ScaleSuffix = ".weight_scale";

    /// <summary>The checkpoint tensors with MXFP4 pairs replaced by MXFP4 weights; <paramref name="tensors"/> itself when there are none.</summary>
    /// <exception cref="InvalidDataException">A packed weight has no scale tensor, or their shapes disagree.</exception>
    public static IReadOnlyDictionary<string, SafetensorsTensor> Resolve(HfConfig config, IReadOnlyDictionary<string, SafetensorsTensor> tensors)
    {
        if (!IsMxfp4Pack(config) || !tensors.Keys.Any(k => k.EndsWith(PackedSuffix, StringComparison.Ordinal)))
        {
            return tensors;
        }

        var result = new Dictionary<string, SafetensorsTensor>(StringComparer.Ordinal);
        foreach (var (name, tensor) in tensors)
        {
            if (name.EndsWith(ScaleSuffix, StringComparison.Ordinal) && tensors.ContainsKey(name[..^ScaleSuffix.Length] + PackedSuffix))
            {
                continue;   // consumed with its packed weight
            }

            if (!name.EndsWith(PackedSuffix, StringComparison.Ordinal))
            {
                result[name] = tensor;
                continue;
            }

            string stem = name[..^PackedSuffix.Length];
            if (!tensors.TryGetValue(stem + ScaleSuffix, out var scale))
            {
                throw new InvalidDataException($"{name}: MXFP4 weight without its scale tensor {stem}{ScaleSuffix}.");
            }

            var weight = Convert(stem + ".weight", tensor, scale);
            result[weight.Name] = new SafetensorsTensor
            {
                Name = weight.Name,
                DType = tensor.DType,
                Type = DataType.MXFP4,
                Shape = [.. weight.Shape.Dims.Select(d => d.Fixed)],
                Offset = tensor.Offset,
                ByteLength = weight.ByteSize,
                File = tensor.File,
                Converted = weight,
            };
        }

        return result;
    }

    /// <summary>Whether the checkpoint declares the compressed-tensors MXFP4 pack format (globally or in a config group).</summary>
    private static bool IsMxfp4Pack(HfConfig config)
    {
        if (!config.Root.TryGetProperty("quantization_config", out var quantization) || quantization.ValueKind != JsonValueKind.Object)
        {
            return false;
        }

        if (quantization.TryGetProperty("format", out var format) && format.GetString() == "mxfp4-pack-quantized")
        {
            return true;
        }

        return quantization.TryGetProperty("config_groups", out var groups) && groups.ValueKind == JsonValueKind.Object
            && groups.EnumerateObject().Any(g => g.Value.ValueKind == JsonValueKind.Object && g.Value.TryGetProperty("format", out var f)
                                                 && f.GetString() == "mxfp4-pack-quantized");
    }

    /// <summary>The MXFP4 weight [out, in] of one packed / scale pair (viewed as a single expert of the gpt-oss block layout).</summary>
    private static WeightInfo Convert(string name, SafetensorsTensor packed, SafetensorsTensor scale)
    {
        if (packed.Shape.Length != 2 || scale.Shape.Length != 2 || scale.Shape[0] != packed.Shape[0] || packed.Shape[1] != scale.Shape[1] * 16)
        {
            throw new InvalidDataException(
                $"{name}: MXFP4 packed [{string.Join(", ", packed.Shape)}] does not match scale [{string.Join(", ", scale.Shape)}].");
        }

        long rows = packed.Shape[0];
        long groups = scale.Shape[1];
        var codes = packed.ToWeightInfo();
        var exponents = scale.ToWeightInfo();
        var blocks = new WeightInfo { Name = codes.Name, Type = DataType.U8, Shape = new Shape(1, rows, groups, 16), Source = codes.Source };
        var scales = new WeightInfo { Name = exponents.Name, Type = DataType.U8, Shape = new Shape(1, rows, groups), Source = exponents.Source };
        var single = HfExpertLayouts.Mxfp4(blocks, scales, name);
        return new WeightInfo { Name = name, Type = DataType.MXFP4, Shape = new Shape(rows, groups * 32), Source = single.Source };
    }
}
