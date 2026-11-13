using System.Collections.Concurrent;

namespace Sharplight.Models;

/// <summary>Global adapter registry; adapters register themselves (or are registered by the host).</summary>
public static class ModelAdapterRegistry
{
    private static readonly ConcurrentDictionary<string, IModelAdapter> s_adapters = new(StringComparer.OrdinalIgnoreCase);

    public static void Register(IModelAdapter adapter)
    {
        foreach (var a in adapter.Architectures)
        {
            s_adapters[a] = adapter;
        }
    }

    public static IModelAdapter? Find(ModelSource source)
        => s_adapters.TryGetValue(source.Architecture, out var a) && a.CanLoad(source) ? a
           : s_adapters.Values.Distinct().FirstOrDefault(x => x.CanLoad(source));

    public static IEnumerable<IModelAdapter> All => s_adapters.Values.Distinct();

    public static ModelDefinition Build(ModelSourceSet sources, AdapterOptions? options = null)
    {
        if (sources.Primary.Definition is { } prebuilt)
        {
            if (sources.Auxiliary.Count > 0)
            {
                throw new NotSupportedException($"'{sources.Primary.Path}' already contains a built model; auxiliary files ({string.Join(", ", sources.Auxiliary.Keys)}) must be added before it is saved.");
            }

            return prebuilt;
        }
        var adapter = Find(sources.Primary)
            ?? throw new NotSupportedException($"No adapter for architecture '{sources.Primary.Architecture}'. Registered: {string.Join(", ", s_adapters.Keys)}");
        return adapter.Build(sources, options ?? new AdapterOptions());
    }
}
