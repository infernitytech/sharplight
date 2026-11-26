using Sharplight.Models;

namespace Sharplight.Package;

/// <summary>
/// Encodes the WACT section: <c>i32 count</c>, then per weight with an activation scheme its name and <c>i32</c>
/// <see cref="WeightInfo.Activations"/>. Readers that do not know the section skip it (the weights then read as
/// <see cref="ActivationScheme.Unspecified"/>).
/// </summary>
internal static class WeightActivationCodec
{
    /// <summary>True when one of <paramref name="weights"/> has an activation scheme to store.</summary>
    public static bool Applies(IEnumerable<WeightInfo> weights) => weights.Any(w => w.Activations != ActivationScheme.Unspecified);

    /// <summary>Writes the schemes of <paramref name="weights"/>.</summary>
    public static void Write(BinaryWriter w, IReadOnlyList<WeightInfo> weights)
    {
        var entries = weights.Where(x => x.Activations != ActivationScheme.Unspecified).ToList();
        w.Write(entries.Count);
        foreach (var e in entries)
        {
            MetadataCodec.WriteString(w, e.Name);
            w.Write((int)e.Activations);
        }
    }

    /// <summary>Replaces each weight of <paramref name="weights"/> the section names by one with its scheme.</summary>
    /// <exception cref="InvalidDataException">An entry has an unknown scheme.</exception>
    public static void Apply(BinaryReader r, Dictionary<string, WeightInfo> weights)
    {
        int n = r.ReadInt32();
        for (int i = 0; i < n; i++)
        {
            string name = MetadataCodec.ReadString(r);
            var scheme = (ActivationScheme)r.ReadInt32();
            if (!Enum.IsDefined(scheme) || scheme == ActivationScheme.Unspecified)
            {
                throw new InvalidDataException($"Weight {name}: activation scheme {(int)scheme}.");
            }

            if (weights.TryGetValue(name, out var w))
            {
                weights[name] = w.WithFacts(w.ScaleRowBlock, scheme);
            }
        }
    }
}
