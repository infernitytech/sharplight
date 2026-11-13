using System.Collections.Concurrent;

namespace Sharplight.Vision;

/// <summary>Registry of <see cref="IMultimodalArchitecture"/>s by canonical architecture id.</summary>
public static class MultimodalRegistry
{
    private static readonly ConcurrentDictionary<string, IMultimodalArchitecture> s_byArch = new(StringComparer.OrdinalIgnoreCase);

    public static void Register(IMultimodalArchitecture m)
    {
        foreach (var a in m.Architectures)
        {
            s_byArch[a] = m;
        }
    }

    public static IMultimodalArchitecture? Find(string architecture) => s_byArch.GetValueOrDefault(architecture);
}
