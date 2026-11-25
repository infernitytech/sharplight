using Sharplight.Runtime;

namespace Sharplight.Models;

/// <summary>
/// The one load path of every model kind: opens the file set through the registered importers (<see cref="Importers"/>), then builds
/// the definition through the registered architectures (<see cref="Architectures"/>), or takes the prebuilt definition of an
/// <c>.slm</c> package. <c>TaskModel</c>, <c>LlmModel</c>, <c>SpanScorer</c> and the package converter all load
/// through a loader. Registered as a singleton by <c>services.AddSharplight()</c> and by every importer or architecture registration;
/// thread-safe.
/// </summary>
/// <example>
/// <code>
/// using var services = SharplightDefaults.CreateServiceProvider();   // Sharplight
/// var loader = services.GetRequiredService&lt;ModelLoader&gt;();
/// using var model = LlmModel.Load(loader, "qwen3-4b-q4_k_m.gguf", new LlmLoadOptions { Devices = [slot] });
/// </code>
/// </example>
public sealed class ModelLoader
{
    /// <summary>Creates a loader over a set of formats and architectures.</summary>
    /// <param name="importers">The model formats.</param>
    /// <param name="architectures">The model architectures.</param>
    public ModelLoader(ModelImporterCatalog importers, ArchitectureCatalog architectures)
    {
        ArgumentNullException.ThrowIfNull(importers);
        ArgumentNullException.ThrowIfNull(architectures);
        Importers = importers;
        Architectures = architectures;
    }

    /// <summary>The model formats this loader opens.</summary>
    public ModelImporterCatalog Importers { get; }

    /// <summary>The model architectures this loader builds.</summary>
    public ArchitectureCatalog Architectures { get; }

    /// <summary>Opens a primary model plus auxiliary sources (see <see cref="ModelImporterCatalog.OpenSet"/>).</summary>
    /// <param name="path">The primary model file or directory.</param>
    /// <param name="auxiliary">Auxiliary files by role, or <see langword="null"/> for none.</param>
    /// <returns>The opened sources.</returns>
    /// <exception cref="NotSupportedException">No registered importer can open one of the paths.</exception>
    public ModelSourceSet Open(string path, IEnumerable<KeyValuePair<string, string>>? auxiliary = null) => Importers.OpenSet(path, auxiliary);

    /// <summary>Builds the definition of opened sources (see <see cref="ArchitectureCatalog.Build"/>).</summary>
    /// <param name="sources">The sources to build from.</param>
    /// <param name="options">Adapter options, or <see langword="null"/> for the defaults.</param>
    /// <param name="log">Receives progress messages, or <see langword="null"/>.</param>
    /// <returns>The built definition.</returns>
    /// <exception cref="NotSupportedException">No adapter handles the architecture.</exception>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions? options = null, Action<string>? log = null)
        => Architectures.Build(sources, options, log);

    /// <summary>Opens and builds a model in one step (see <see cref="Open"/> and <see cref="Build(ModelSourceSet, AdapterOptions?, Action{string}?)"/>).</summary>
    /// <param name="path">The primary model file or directory.</param>
    /// <param name="auxiliary">Auxiliary files by role, or <see langword="null"/> for none.</param>
    /// <param name="options">Adapter options, or <see langword="null"/> for the defaults.</param>
    /// <returns>The built definition.</returns>
    /// <exception cref="NotSupportedException">No registered importer or adapter handles the model.</exception>
    public ModelDefinition Build(string path, IEnumerable<KeyValuePair<string, string>>? auxiliary = null, AdapterOptions? options = null)
    {
        var sources = Open(path, auxiliary);
        try
        {
            return Build(sources, options);
        }
        catch
        {
            DisposeAll(sources);
            throw;
        }
    }

    /// <summary>Opens and builds a model, reporting the reading and building stages.</summary>
    /// <param name="path">Model file or directory.</param>
    /// <param name="auxiliary">Auxiliary files by role.</param>
    /// <param name="adapter">Adapter settings.</param>
    /// <param name="progress">Load progress receiver, or <see langword="null"/>.</param>
    /// <param name="log">Log receiver, or <see langword="null"/>.</param>
    /// <param name="cancellationToken">Cancels between the stages.</param>
    /// <param name="transformSources">Rewrites the opened sources before the build (e.g. a lazy weight conversion), or <see langword="null"/>.</param>
    /// <returns>The built definition and the primary source it was built from.</returns>
    internal (ModelDefinition Definition, ModelSource Source) Load(string path, IEnumerable<KeyValuePair<string, string>> auxiliary,
        AdapterOptions adapter, IProgress<LoadProgress>? progress, Action<string>? log, CancellationToken cancellationToken,
        Func<ModelSourceSet, ModelSourceSet>? transformSources = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        cancellationToken.ThrowIfCancellationRequested();
        var reading = new LoadProgressCounter(progress, LoadStage.Reading, 1, path);
        var sources = Open(path, auxiliary);
        reading.Complete();
        try
        {
            if (transformSources is not null)
            {
                sources = transformSources(sources);
            }

            cancellationToken.ThrowIfCancellationRequested();
            var building = new LoadProgressCounter(progress, LoadStage.Building, 1);
            var definition = Build(sources, adapter, log);
            building.Complete();
            return (definition, sources.Primary);
        }
        catch
        {
            DisposeAll(sources);
            throw;
        }
    }

    private static void DisposeAll(ModelSourceSet sources)
    {
        sources.Primary.DisposeResources();
        foreach (var source in sources.Auxiliary.Values)
        {
            source.DisposeResources();
        }
    }
}
