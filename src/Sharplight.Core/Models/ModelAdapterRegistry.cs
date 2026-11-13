using System.Collections.Concurrent;
using Sharplight.Graph.Optimization;

namespace Sharplight.Models;

/// <summary>Global adapter registry; adapters register themselves (or are registered by the host).</summary>
public static class ModelAdapterRegistry
{
    private static readonly ConcurrentDictionary<string, IModelAdapter> s_adapters = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// Registers an adapter for each of its architectures; a later registration for the same architecture replaces the earlier one.
    /// </summary>
    public static void Register(IModelAdapter adapter)
    {
        foreach (var a in adapter.Architectures)
        {
            s_adapters[a] = adapter;
        }
    }

    /// <summary>
    /// Finds the adapter for a source: the one registered for its architecture, else any adapter whose <see cref="IModelAdapter.CanLoad"/> accepts it.
    /// </summary>
    /// <returns>The adapter, or <see langword="null"/> when none matches.</returns>
    public static IModelAdapter? Find(ModelSource source)
        => s_adapters.TryGetValue(source.Architecture, out var a) && a.CanLoad(source) ? a
           : s_adapters.Values.Distinct().FirstOrDefault(x => x.CanLoad(source));

    /// <summary>The registered adapters, each once.</summary>
    public static IEnumerable<IModelAdapter> All => s_adapters.Values.Distinct();

    /// <summary>
    /// Builds the model definition for a set of sources: a prebuilt definition (saved package) is used as is, otherwise the matching adapter
    /// builds it. Then the graph optimization phase runs unless the definition already carries the current pipeline id or
    /// <see cref="AdapterOptions.OptimizeGraphs"/> is false (see <see cref="GraphOptimizer.PrepareForLoad"/>).
    /// </summary>
    /// <param name="sources">The sources to build from.</param>
    /// <param name="options">Adapter options, or <see langword="null"/> for the defaults.</param>
    /// <param name="log">Receives progress messages (the optimization summary), or <see langword="null"/>.</param>
    /// <exception cref="NotSupportedException">No adapter handles the architecture, or auxiliary sources were given with an already built definition.</exception>
    public static ModelDefinition Build(ModelSourceSet sources, AdapterOptions? options = null, Action<string>? log = null)
    {
        options ??= new AdapterOptions();
        if (sources.Primary.Definition is { } prebuilt)
        {
            if (sources.Auxiliary.Count > 0)
            {
                throw new NotSupportedException($"'{sources.Primary.Path}' already contains a built model; auxiliary files ({string.Join(", ", sources.Auxiliary.Keys)}) must be added before it is saved.");
            }

            return GraphOptimizer.PrepareForLoad(prebuilt, options.OptimizeGraphs, log);
        }
        var adapter = Find(sources.Primary)
            ?? throw new NotSupportedException($"No adapter for architecture '{sources.Primary.Architecture}'. Registered: {string.Join(", ", s_adapters.Keys)}");
        return GraphOptimizer.PrepareForLoad(adapter.Build(sources, options), options.OptimizeGraphs, log);
    }
}
