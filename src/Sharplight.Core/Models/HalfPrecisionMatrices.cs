using System.Runtime.InteropServices;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Models;

/// <summary>
/// Stores the F32 matrices of a model as F16 at load (<see cref="Tasks.TaskLoadOptions.HalfPrecisionMatrices"/>): rank ≥ 2 F32 weights
/// are converted lazily when they are uploaded; norms, biases, quantized and 16-bit weights and prebuilt (package) definitions are kept.
/// </summary>
internal static class HalfPrecisionMatrices
{
    /// <summary>Whether <paramref name="setting"/> turns the conversion on for <paramref name="devices"/> (null = on when the first device is a GPU).</summary>
    public static bool Enabled(bool? setting, IEnumerable<DeviceSlot> devices)
        => setting ?? devices.FirstOrDefault()?.Device.Info.Kind is not (null or DeviceKind.Cpu);

    /// <summary>The source transform of <see cref="ModelLoader"/> that converts the primary source's matrices.</summary>
    public static ModelSourceSet Transform(ModelSourceSet sources) => new() { Primary = Convert(sources.Primary), Auxiliary = sources.Auxiliary };

    private static ModelSource Convert(ModelSource s)
    {
        if (s.Definition is not null || !s.Weights.Values.Any(IsHalved))
        {
            return s;
        }

        var weights = s.Weights.ToDictionary(kv => kv.Key, kv => IsHalved(kv.Value) ? ToHalf(kv.Value) : kv.Value);
        return new ModelSource
        {
            Format = s.Format, Architecture = s.Architecture, Metadata = s.Metadata, Weights = weights, OwnedResources = s.OwnedResources,
            FormatData = s.FormatData, Path = s.Path, CompilationCaches = s.CompilationCaches,
        };
    }

    // F32 matrices, except the short-conv kernels of recurrent layers (decision models on hybrid trunks), which backends take in f32.
    private static bool IsHalved(WeightInfo w) => w.Type == DataType.F32 && w.Shape.Rank >= 2 && !w.Name.EndsWith(".conv1d.weight", StringComparison.Ordinal);

    /// <summary>Lazily converted copy (read when the weight is uploaded).</summary>
    private static WeightInfo ToHalf(WeightInfo w) => WeightInfo.Derived(w.Name, DataType.F16, w.Shape, () =>
    {
        var src = MemoryMarshal.Cast<byte, float>(w.ReadAll());
        var dst = new byte[src.Length * 2];
        var h = MemoryMarshal.Cast<byte, Half>(dst);
        for (int i = 0; i < src.Length; i++)
        {
            h[i] = (Half)src[i];
        }

        return dst;
    });
}
