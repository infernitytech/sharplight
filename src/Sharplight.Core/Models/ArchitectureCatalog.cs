using System.Diagnostics.CodeAnalysis;
using Sharplight.Graph.Optimization;
using Sharplight.Vision;

namespace Sharplight.Models;

/// <summary>
/// The model architectures a host supports: the <see cref="IModelAdapter"/>, <see cref="IMultimodalArchitecture"/> and
/// <see cref="AuxiliaryArchitecture"/> services registered with the container (<c>services.AddBuiltinArchitectures()</c>, or
/// <c>services.AddModelAdapter&lt;T&gt;()</c> and friends). Registered as a singleton by every architecture registration; thread-safe.
/// </summary>
public sealed class ArchitectureCatalog
{
    private readonly IModelAdapter[] _adapters;
    private readonly ICompositeModelAdapter[] _composites;
    private readonly Dictionary<string, IModelAdapter> _byArchitecture = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, IMultimodalArchitecture> _multimodal = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, string> _auxiliaryOnly = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>Creates the catalog.</summary>
    /// <param name="adapters">The adapters in registration order; for an architecture claimed by several, the last one wins.</param>
    /// <param name="multimodal">Prompt assembly for image-capable architectures; for an architecture claimed by several, the last one wins.</param>
    /// <param name="auxiliaryOnly">Architectures that are only loaded as auxiliary sources.</param>
    /// <param name="composites">Adapters that build one model from several sources, consulted in registration order.</param>
    public ArchitectureCatalog(IEnumerable<IModelAdapter> adapters, IEnumerable<IMultimodalArchitecture>? multimodal = null,
        IEnumerable<AuxiliaryArchitecture>? auxiliaryOnly = null, IEnumerable<ICompositeModelAdapter>? composites = null)
    {
        ArgumentNullException.ThrowIfNull(adapters);
        _adapters = [.. adapters.Distinct()];
        _composites = [.. (composites ?? []).Distinct()];
        foreach (var adapter in _adapters)
        {
            foreach (var architecture in adapter.Architectures)
            {
                _byArchitecture[architecture] = adapter;
            }
        }

        foreach (var handler in multimodal ?? [])
        {
            foreach (var architecture in handler.Architectures)
            {
                _multimodal[architecture] = handler;
            }
        }

        foreach (var auxiliary in auxiliaryOnly ?? [])
        {
            _auxiliaryOnly[auxiliary.Architecture] = auxiliary.Role;
        }
    }

    /// <summary>The registered adapters, each once, in registration order.</summary>
    public IReadOnlyList<IModelAdapter> Adapters => _adapters;

    /// <summary>The registered composite adapters, each once, in registration order.</summary>
    public IReadOnlyList<ICompositeModelAdapter> CompositeAdapters => _composites;

    /// <summary>
    /// Finds the adapter for a source: the one registered for its architecture, else the first adapter whose
    /// <see cref="IModelAdapter.CanLoad"/> accepts it.
    /// </summary>
    /// <param name="source">The primary source.</param>
    /// <param name="adapter">The adapter, or <see langword="null"/> when none matches.</param>
    /// <returns><see langword="true"/> when an adapter was found.</returns>
    public bool TryFindAdapter(ModelSource source, [NotNullWhen(true)] out IModelAdapter? adapter)
    {
        ArgumentNullException.ThrowIfNull(source);
        adapter = _byArchitecture.TryGetValue(source.Architecture, out var registered) && registered.CanLoad(source)
            ? registered
            : _adapters.FirstOrDefault(a => a.CanLoad(source));
        return adapter is not null;
    }

    /// <summary>Finds the multimodal prompt assembly of a canonical architecture.</summary>
    /// <param name="architecture">The canonical architecture id.</param>
    /// <param name="multimodal">The handler, or <see langword="null"/> when the architecture has no media support.</param>
    /// <returns><see langword="true"/> when a handler was found.</returns>
    public bool TryFindMultimodal(string architecture, [NotNullWhen(true)] out IMultimodalArchitecture? multimodal)
    {
        ArgumentNullException.ThrowIfNull(architecture);
        return _multimodal.TryGetValue(architecture, out multimodal);
    }

    /// <summary>
    /// Builds the model definition for a set of sources: a prebuilt definition (saved package) is used as is, otherwise the matching adapter
    /// builds it (a composite adapter that accepts the whole set first, see <see cref="ICompositeModelAdapter"/>). Then the graph optimization phase runs unless the definition already carries the current pipeline id or
    /// <see cref="AdapterOptions.OptimizeGraphs"/> is false (see <see cref="GraphOptimizer.PrepareForLoad"/>).
    /// </summary>
    /// <param name="sources">The sources to build from.</param>
    /// <param name="options">Adapter options, or <see langword="null"/> for the defaults.</param>
    /// <param name="log">Receives progress messages (the optimization summary), or <see langword="null"/>.</param>
    /// <returns>The built definition; it takes over the weights of <paramref name="sources"/>.</returns>
    /// <exception cref="NotSupportedException">No adapter handles the architecture, or auxiliary sources were given with an already built definition.</exception>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions? options = null, Action<string>? log = null)
    {
        ArgumentNullException.ThrowIfNull(sources);
        options ??= new AdapterOptions();
        if (sources.Primary.Definition is { } prebuilt)
        {
            if (sources.Auxiliary.Count > 0)
            {
                throw new NotSupportedException($"'{sources.Primary.Path}' already contains a built model; auxiliary files ({string.Join(", ", sources.Auxiliary.Keys)}) must be added before it is saved.");
            }

            return GraphOptimizer.PrepareForLoad(prebuilt, options.OptimizeGraphs, log);
        }

        if (sources.Auxiliary.Count > 0 && _composites.FirstOrDefault(c => c.CanLoad(sources)) is { } composite)
        {
            return GraphOptimizer.PrepareForLoad(composite.Build(sources, options, this), options.OptimizeGraphs, log);
        }

        if (!TryFindAdapter(sources.Primary, out var adapter))
        {
            throw NoAdapter(sources.Primary);
        }

        return GraphOptimizer.PrepareForLoad(adapter.Build(sources, options), options.OptimizeGraphs, log);
    }

    private NotSupportedException NoAdapter(ModelSource source)
    {
        if (_auxiliaryOnly.TryGetValue(source.Architecture, out var role))
        {
            return new NotSupportedException($"'{source.Path}' is a {role} model (architecture '{source.Architecture}'), not a model of its own: load it as the "
                + $"'{role}' auxiliary source of the model it was made for.");
        }

        if (_adapters.Length == 0)
        {
            return new NotSupportedException($"No adapter for architecture '{source.Architecture}': no architecture adapters are registered. Register them "
                + "with services.AddSharplight() (Sharplight.Defaults) or services.AddBuiltinArchitectures() (Sharplight.Architectures).");
        }

        return new NotSupportedException($"No adapter for architecture '{source.Architecture}'. Registered: {string.Join(", ", _byArchitecture.Keys)}");
    }
}
