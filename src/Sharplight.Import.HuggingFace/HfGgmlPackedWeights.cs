using System.Text.Json;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// ggml blocks stored in safetensors (NVIDIA ModelOpt <c>MIXED_PRECISION</c> checkpoints, e.g. Nemotron-3.5-Bolt-74B-A9B-24GB): a
/// <c>quantization_config.config_groups</c> entry with <c>"packing": "ggml"</c> names the block format (<c>quant_algo</c> Q8_0, IQ2_XXS, ...),
/// its <c>group_size</c> and <c>block_payload_bytes</c>, and the modules stored that way (<c>targets</c>); each such <c>X.weight</c> is U8
/// [N, K / group_size, block_payload_bytes], the rows of the matching block type laid out as in a GGUF file. They become weights of that
/// type over the same bytes, without a copy.
/// </summary>
internal static class HfGgmlPackedWeights
{
    private const string GgmlPacking = "ggml";

    /// <summary>The tensors with every ggml-packed weight viewed as its block type; <paramref name="tensors"/> itself when there are none.</summary>
    /// <exception cref="NotSupportedException">A group names a block format this version does not have.</exception>
    /// <exception cref="InvalidDataException">A packed tensor does not have the group's block size.</exception>
    public static IReadOnlyDictionary<string, SafetensorsTensor> Resolve(HfConfig config, IReadOnlyDictionary<string, SafetensorsTensor> tensors)
    {
        var formats = TargetFormats(config);
        if (formats.Count == 0)
        {
            return tensors;
        }

        var result = new Dictionary<string, SafetensorsTensor>(tensors, StringComparer.Ordinal);
        foreach (var (module, type) in formats)
        {
            if (!tensors.TryGetValue(module + ".weight", out var packed))
            {
                continue;
            }

            if (packed.DType != "U8" || packed.Shape.Length != 3 || packed.Shape[2] != type.BlockBytes)
            {
                throw new InvalidDataException(
                    $"{packed.Name}: [{string.Join(", ", packed.Shape)}] {packed.DType} is not U8 [N, K / {type.BlockSize}, {type.BlockBytes}] {type.Name} blocks.");
            }

            long[] shape = [packed.Shape[0], packed.Shape[1] * type.BlockSize];
            var weight = new WeightInfo
            {
                Name = packed.Name,
                Type = type,
                Shape = new Shape(shape.Select(d => (Dim)d)),
                Source = new SafetensorsWeightSource(packed.File, packed.File.DataOffset + packed.Offset, packed.ByteLength),
            };
            result[packed.Name] = new SafetensorsTensor
            {
                Name = packed.Name,
                DType = packed.DType,
                Type = type,
                Shape = shape,
                Offset = packed.Offset,
                ByteLength = packed.ByteLength,
                File = packed.File,
                Converted = weight,
            };
        }

        return result;
    }

    // Module name -> block type of every ggml-packed config group.
    private static Dictionary<string, DataType> TargetFormats(HfConfig config)
    {
        var formats = new Dictionary<string, DataType>(StringComparer.Ordinal);
        if (!config.TryGetQuantization("quantization_config", out var quantization)
            || !quantization.TryGetProperty("config_groups", out var groups) || groups.ValueKind != JsonValueKind.Object)
        {
            return formats;
        }

        foreach (var group in groups.EnumerateObject())
        {
            var g = group.Value;
            if (g.ValueKind != JsonValueKind.Object || !g.TryGetProperty("packing", out var packing) || packing.GetString() != GgmlPacking)
            {
                continue;
            }

            string algorithm = g.GetProperty("quant_algo").GetString() ?? "";
            var type = DataType.TryGet(algorithm.ToLowerInvariant());
            if (type is null || !type.IsQuantized
                || (g.TryGetProperty("group_size", out var size) && size.GetInt32() != type.BlockSize)
                || (g.TryGetProperty("block_payload_bytes", out var bytes) && bytes.GetInt32() != type.BlockBytes))
            {
                throw new NotSupportedException($"ggml-packed weights of format '{algorithm}' are not supported.");
            }

            foreach (var target in g.GetProperty("targets").EnumerateArray())
            {
                formats[target.GetString()!] = type;
            }
        }

        return formats;
    }
}
