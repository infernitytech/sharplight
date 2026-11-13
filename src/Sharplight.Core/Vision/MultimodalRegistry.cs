using System.Collections.Concurrent;

namespace Sharplight.Vision;

/// <summary>Registry of <see cref="IMultimodalArchitecture"/>s by canonical architecture id.</summary>
public static class MultimodalRegistry
{
    private static readonly ConcurrentDictionary<string, IMultimodalArchitecture> s_byArch = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>Registers a handler for each of its architectures; a later registration replaces an earlier one.</summary>
    public static void Register(IMultimodalArchitecture m)
    {
        foreach (var a in m.Architectures)
        {
            s_byArch[a] = m;
        }
    }

    /// <summary>
    /// Gets the handler for a canonical architecture id, or <see langword="null"/> when the model has no media support.
    /// </summary>
    public static IMultimodalArchitecture? Find(string architecture) => s_byArch.GetValueOrDefault(architecture);
}
