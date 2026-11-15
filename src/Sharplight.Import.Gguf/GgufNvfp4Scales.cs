using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.Gguf;

/// <summary>
/// NVFP4 tensors that carry a separate output multiplier (<c>X.weight</c> NVFP4 with <c>X.scale</c> of one value per tensor or per
/// expert, as in ModelOpt-derived files such as LFM3 NVFP4 and the Nemotron-3.5 NVFP4 files): each becomes one
/// <see cref="DataType.NVFP4_GS"/> weight (<see cref="GgufNvfp4ScaledSource"/>) before the architecture mapping sees it, so every
/// mapping, split and expert stack works on it unchanged. The multiplier and the static activation scale <c>X.input_scale</c>
/// (activations are not quantized) are consumed.
/// </summary>
internal static class GgufNvfp4Scales
{
    private const string WeightSuffix = ".weight";
    private const string ScaleSuffix = ".scale";
    private const string InputScaleSuffix = ".input_scale";

    /// <summary>The converted weights by tensor name, and the names of the tensors they consume.</summary>
    /// <param name="tensors">Every tensor of every shard.</param>
    /// <exception cref="InvalidDataException">A multiplier's element count matches neither the tensor nor its matrices.</exception>
    public static (Dictionary<string, WeightInfo> Converted, HashSet<string> Consumed) Find(IReadOnlyList<GgufTensorInfo> tensors)
    {
        var converted = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
        var consumed = new HashSet<string>(StringComparer.Ordinal);
        var byName = new Dictionary<string, GgufTensorInfo>(StringComparer.Ordinal);
        foreach (var tensor in tensors)
        {
            byName.TryAdd(tensor.Name, tensor);
        }

        foreach (var tensor in tensors)
        {
            if (tensor.DataType != DataType.NVFP4 || !tensor.Name.EndsWith(WeightSuffix, StringComparison.Ordinal))
            {
                continue;
            }

            string stem = tensor.Name[..^WeightSuffix.Length];
            if (!byName.TryGetValue(stem + ScaleSuffix, out var scale)
                || !(scale.DataType == DataType.F32 || scale.DataType == DataType.F16 || scale.DataType == DataType.BF16))
            {
                continue;
            }

            long columns = tensor.Ne[0];
            long rows = tensor.Ne.Length > 1 ? tensor.Ne[1] : 1;
            long matrices = tensor.Ne.Skip(2).Aggregate(1L, (a, d) => a * d);
            long multipliers = scale.Ne.Aggregate(1L, (a, d) => a * d);
            if (multipliers != 1 && multipliers != matrices)
            {
                throw new InvalidDataException($"{scale.Name}: {multipliers} multipliers for {matrices} NVFP4 matrices of {tensor.Name}.");
            }

            var codes = tensor.ToWeightInfo();
            converted[tensor.Name] = new WeightInfo
            {
                Name = codes.Name,
                Type = DataType.NVFP4_GS,
                Shape = codes.Shape,
                Source = new GgufNvfp4ScaledSource(codes.Source, scale.ToWeightInfo(), rows * columns / DataType.NVFP4.BlockSize),
            };
            consumed.Add(scale.Name);
            if (byName.ContainsKey(stem + InputScaleSuffix))
            {
                consumed.Add(stem + InputScaleSuffix);
            }
        }

        return (converted, consumed);
    }
}
