using System.Collections.Concurrent;
using Sharplight.Graph.Optimization;

namespace Sharplight.Models;

/// <summary>Global adapter registry; adapters register themselves (or are registered by the host).</summary>
public static class ModelAdapterRegistry
{
    private static readonly ConcurrentDictionary<string, IModelAdapter> s_adapters = new(StringComparer.OrdinalIgnoreCase);
    private static readonly ConcurrentDictionary<string, string> s_auxiliaryOnly = new(StringComparer.OrdinalIgnoreCase);

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
    /// Registers a canonical architecture whose models are never loaded on their own but only as an auxiliary source of another model (e.g. a
    /// speculative-decoding drafter as the <c>draft</c> source of its target), so loading one on its own fails with a message that says so.
    /// </summary>
    /// <param name="architecture">The canonical architecture id.</param>
    /// <param name="role">The auxiliary role its models are loaded in (e.g. <c>draft</c>).</param>
    public static void RegisterAuxiliaryOnly(string architecture, string role) => s_auxiliaryOnly[architecture] = role;

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
        var adapter = Find(sources.Primary) ?? throw NoAdapter(sources.Primary);
        return GraphOptimizer.PrepareForLoad(adapter.Build(sources, options), options.OptimizeGraphs, log);
    }

    private static NotSupportedException NoAdapter(ModelSource source)
        => s_auxiliaryOnly.TryGetValue(source.Architecture, out var role)
            ? new NotSupportedException($"'{source.Path}' is a {role} model (architecture '{source.Architecture}'), not a model of its own: load it as the "
                + $"'{role}' auxiliary source of the model it was made for.")
            : s_adapters.IsEmpty
                ? new NotSupportedException($"No adapter for architecture '{source.Architecture}': no architecture adapters are registered. Register them at startup "
                    + "with SharplightDefaults.Register() (Sharplight.Defaults package) or BuiltinAdapters.RegisterAll() (Sharplight.Architectures).")
                : new NotSupportedException($"No adapter for architecture '{source.Architecture}'. Registered: {string.Join(", ", s_adapters.Keys)}");
}
