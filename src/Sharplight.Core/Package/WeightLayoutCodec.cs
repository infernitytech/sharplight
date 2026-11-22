using Sharplight.Models;

namespace Sharplight.Package;

/// <summary>
/// Encodes the WLAY section: <c>i32 count</c>, then per weight with a layout fact its name and <c>i32</c>
/// <see cref="WeightInfo.ScaleRowBlock"/>. Readers that do not know the section skip it (the weights then have no such facts).
/// </summary>
internal static class WeightLayoutCodec
{
    /// <summary>True when one of <paramref name="weights"/> has a layout fact to store.</summary>
    public static bool Applies(IEnumerable<WeightInfo> weights) => weights.Any(w => w.ScaleRowBlock > 0);

    /// <summary>Writes the facts of <paramref name="weights"/>.</summary>
    public static void Write(BinaryWriter w, IReadOnlyList<WeightInfo> weights)
    {
        var entries = weights.Where(x => x.ScaleRowBlock > 0).ToList();
        w.Write(entries.Count);
        foreach (var e in entries)
        {
            MetadataCodec.WriteString(w, e.Name);
            w.Write(e.ScaleRowBlock);
        }
    }

    /// <summary>Replaces each weight of <paramref name="weights"/> the section names by one with its facts.</summary>
    /// <exception cref="InvalidDataException">An entry has a non-positive row block.</exception>
    public static void Apply(BinaryReader r, Dictionary<string, WeightInfo> weights)
    {
        int n = r.ReadInt32();
        for (int i = 0; i < n; i++)
        {
            string name = MetadataCodec.ReadString(r);
            int block = r.ReadInt32();
            if (block <= 0)
            {
                throw new InvalidDataException($"Weight {name}: scale row block {block}.");
            }

            if (weights.TryGetValue(name, out var w))
            {
                weights[name] = new WeightInfo { Name = w.Name, Type = w.Type, Shape = w.Shape, Source = w.Source, InputOrder = w.InputOrder, ScaleRowBlock = block };
            }
        }
    }
}
