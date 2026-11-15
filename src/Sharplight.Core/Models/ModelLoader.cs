using Sharplight.Runtime;

namespace Sharplight.Models;

/// <summary>
/// The one load path of every model kind: open the file set through the registered importers, then build the definition through the
/// architecture adapters (or take the prebuilt definition of an <c>.slm</c> package). <c>LlmModel</c>, <c>EmbeddingModel</c> and
/// <c>TaskModel</c> all load through here.
/// </summary>
internal static class ModelLoader
{
    /// <summary>Opens and builds a model, reporting the reading and building stages.</summary>
    /// <param name="path">Model file or directory.</param>
    /// <param name="auxiliary">Auxiliary files by role.</param>
    /// <param name="adapter">Adapter settings.</param>
    /// <param name="progress">Load progress receiver, or <see langword="null"/>.</param>
    /// <param name="log">Log receiver, or <see langword="null"/>.</param>
    /// <param name="cancellationToken">Cancels between the stages.</param>
    /// <param name="transformSources">Rewrites the opened sources before the build (e.g. a lazy weight conversion), or <see langword="null"/>.</param>
    /// <returns>The built definition and the primary source it was built from.</returns>
    public static (ModelDefinition Definition, ModelSource Source) Load(string path, IEnumerable<(string Role, string Path)> auxiliary,
        AdapterOptions adapter, IProgress<LoadProgress>? progress, Action<string>? log, CancellationToken cancellationToken,
        Func<ModelSourceSet, ModelSourceSet>? transformSources = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        cancellationToken.ThrowIfCancellationRequested();
        var reading = new LoadProgressCounter(progress, LoadStage.Reading, 1, path);
        var sources = ModelImporterRegistry.OpenSet(path, [.. auxiliary]);
        reading.Complete();
        if (transformSources is not null)
        {
            sources = transformSources(sources);
        }

        cancellationToken.ThrowIfCancellationRequested();
        var building = new LoadProgressCounter(progress, LoadStage.Building, 1);
        var definition = ModelAdapterRegistry.Build(sources, adapter, log);
        building.Complete();
        return (definition, sources.Primary);
    }
}
