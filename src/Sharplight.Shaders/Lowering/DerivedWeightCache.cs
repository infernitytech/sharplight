using System.Runtime.CompilerServices;
using Sharplight.Models;

namespace Sharplight.Shaders.Lowering;

/// <summary>
/// The weights derived on the host from each model weight (the split-f16 parts and scales of <see cref="SplitScaleTable"/>, the
/// channels-last 16-bit convolution weights), one per source tensor and derivation for the whole process. Every compilation of a
/// model then names the same derived <see cref="IWeightSource"/>, so <see cref="Runtime.SharedWeights"/> uploads it once per device
/// for all of them (a decoder's prompt graph and its draft head, several sessions), as it does for the original tensors. Entries are
/// weak: the derived bytes go away with the last compiled program or shared device buffer that references them.
/// </summary>
internal static class DerivedWeightCache
{
    private static readonly ConditionalWeakTable<IWeightSource, Dictionary<object, WeakReference<IWeightSource>>> Table = new();

    /// <summary>
    /// The derivation <paramref name="key"/> (an equatable description: form, shape, order) of <paramref name="weight"/>.
    /// <paramref name="create"/> makes a new one (a lazy source, nothing computed yet); while an earlier one's source is alive, the
    /// result names that source instead.
    /// </summary>
    public static WeightInfo Get(WeightInfo weight, object key, Func<WeightInfo> create)
    {
        var byKey = Table.GetValue(weight.Source, _ => []);
        lock (byKey)
        {
            var entry = create();
            if (byKey.TryGetValue(key, out var weak) && weak.TryGetTarget(out var source))
            {
                return new WeightInfo { Name = entry.Name, Type = entry.Type, Shape = entry.Shape, Source = source };
            }

            byKey[key] = new WeakReference<IWeightSource>(entry.Source);
            return entry;
        }
    }
}
