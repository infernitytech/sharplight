using System.Runtime.CompilerServices;
using Sharplight.Models;

namespace Sharplight.Shaders.Lowering;

/// <summary>
/// The derived split-f16 weights (<see cref="SplitScaleTable"/>) of each model weight, one pair per source tensor and matrix
/// shape for the whole process. Every compilation of a model then names the same derived <see cref="IWeightSource"/>, so
/// <see cref="Runtime.SharedWeights"/> uploads the split parts once per device for all of them (a decoder's prompt graph and
/// its draft head, several sessions), as it does for the original tensors. Entries are weak: the derived bytes go away with
/// the last compiled program or shared device buffer that references them.
/// </summary>
internal static class SplitWeightCache
{
    private static readonly ConditionalWeakTable<IWeightSource, Dictionary<(long Rows, long K), (WeakReference<IWeightSource> Parts, WeakReference<IWeightSource> Scales)>> Table = new();

    /// <summary>
    /// The split parts and row scales of <paramref name="weight"/> viewed as <paramref name="rows"/> x <paramref name="k"/> f32
    /// values. <paramref name="create"/> makes a new pair (lazy sources, nothing computed yet); while an earlier pair's sources
    /// are alive, the result names those instead.
    /// </summary>
    public static (WeightInfo Parts, WeightInfo Scales) Get(WeightInfo weight, long rows, long k, Func<(WeightInfo Parts, WeightInfo Scales)> create)
    {
        var byShape = Table.GetValue(weight.Source, _ => []);
        lock (byShape)
        {
            var entry = create();
            if (byShape.TryGetValue((rows, k), out var weak) && weak.Parts.TryGetTarget(out var parts) && weak.Scales.TryGetTarget(out var scales))
            {
                return (WithSource(entry.Parts, parts), WithSource(entry.Scales, scales));
            }

            byShape[(rows, k)] = (new WeakReference<IWeightSource>(entry.Parts.Source), new WeakReference<IWeightSource>(entry.Scales.Source));
            return entry;
        }
    }

    private static WeightInfo WithSource(WeightInfo info, IWeightSource source)
        => new() { Name = info.Name, Type = info.Type, Shape = info.Shape, Source = source };
}
