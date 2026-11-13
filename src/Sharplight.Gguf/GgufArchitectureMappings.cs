using System.Collections.Concurrent;

namespace Sharplight.Gguf;

/// <summary>Registry of <see cref="IGgufArchitectureMapping"/>s keyed by GGUF architecture id (case-insensitive).</summary>
internal static class GgufArchitectureMappings
{
    private static readonly ConcurrentDictionary<string, IGgufArchitectureMapping> s_byArch = new(StringComparer.OrdinalIgnoreCase);

    static GgufArchitectureMappings()
    {
        Register(new Qwen35GgufMapping());
        Register(new NemotronHGgufMapping());
        Register(new HybridGgufMapping());
        Register(new Rwkv7GgufMapping());
        Register(new Rwkv6GgufMapping());
        Register(new ClipGgufMapping());
        Register(new DecoderGgufMapping());
        Register(new EncoderGgufMapping());
    }

    /// <summary>Registers <paramref name="mapping"/> for each of its GGUF architectures; later registrations win.</summary>
    public static void Register(IGgufArchitectureMapping mapping)
    {
        foreach (var architecture in mapping.GgufArchitectures)
        {
            s_byArch[architecture] = mapping;
        }
    }

    /// <summary>Every registered mapping once.</summary>
    public static IEnumerable<IGgufArchitectureMapping> All => s_byArch.Values.Distinct();

    /// <summary>The mapping for a <c>general.architecture</c> value, or null.</summary>
    public static IGgufArchitectureMapping? Find(string? ggufArchitecture)
        => ggufArchitecture != null && s_byArch.TryGetValue(ggufArchitecture, out var m) ? m : null;
}
