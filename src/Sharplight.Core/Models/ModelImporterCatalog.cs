using System.Diagnostics.CodeAnalysis;

namespace Sharplight.Models;

/// <summary>
/// The model formats a host supports: the <see cref="IModelImporter"/> services registered with the container
/// (<c>services.AddGgufImporter()</c>, <c>services.AddHuggingFaceImporter()</c>, ... or <c>services.AddModelImporter&lt;T&gt;()</c>).
/// Registered as a singleton by every importer registration; thread-safe.
/// </summary>
public sealed class ModelImporterCatalog
{
    private readonly IModelImporter[] _importers;

    /// <summary>Creates the catalog over a set of importers.</summary>
    /// <param name="importers">The importers in registration order; of several with the same <see cref="IModelImporter.Format"/> the last one wins.</param>
    public ModelImporterCatalog(IEnumerable<IModelImporter> importers)
    {
        ArgumentNullException.ThrowIfNull(importers);
        var all = importers.ToList();
        _importers = [.. all.Where((importer, index) => all.FindLastIndex(other => string.Equals(other.Format, importer.Format, StringComparison.OrdinalIgnoreCase)) == index)];
    }

    /// <summary>The importers, in the order they are tried.</summary>
    public IReadOnlyList<IModelImporter> Importers => _importers;

    /// <summary>
    /// Finds the importer with the highest <see cref="IModelImporter.Priority"/> whose <see cref="IModelImporter.CanOpen"/> accepts
    /// <paramref name="path"/> (ties: by format name). A format that shares a container with another (a diffusion checkpoint in a GGUF or
    /// safetensors file) claims its files by checking their content and ranking above the generic importer.
    /// </summary>
    /// <param name="path">File or directory.</param>
    /// <param name="importer">The importer, or <see langword="null"/> when none accepts the path.</param>
    /// <returns><see langword="true"/> when an importer was found.</returns>
    public bool TryFind(string path, [NotNullWhen(true)] out IModelImporter? importer)
    {
        ArgumentNullException.ThrowIfNull(path);
        importer = _importers.OrderByDescending(i => i.Priority).ThenBy(i => i.Format, StringComparer.Ordinal).FirstOrDefault(i => i.CanOpen(path));
        return importer is not null;
    }

    /// <summary>Opens a model file or directory with the matching importer.</summary>
    /// <param name="path">File or directory to open.</param>
    /// <returns>The canonical source; the caller owns its resources (<see cref="ModelSource.OwnedResources"/>).</returns>
    /// <exception cref="NotSupportedException">No registered importer can open the path.</exception>
    public ModelSource Open(string path)
    {
        if (!TryFind(path, out var importer))
        {
            string formats = _importers.Length == 0 ? "(none; register importers with services.AddSharplight() or services.AddGgufImporter() and friends)"
                : string.Join(", ", _importers.Select(i => i.Format));
            throw new NotSupportedException($"No registered importer can open '{path}'. Registered formats: {formats}");
        }

        return importer.Open(path);
    }

    /// <summary>Opens a primary model plus auxiliary sources (e.g. a vision projector under <see cref="Vision.AuxiliaryRoles.Mmproj"/>).</summary>
    /// <param name="path">The primary model file or directory.</param>
    /// <param name="auxiliary">Auxiliary files by role, or <see langword="null"/> for none.</param>
    /// <returns>The opened sources; on failure everything opened so far is released.</returns>
    /// <exception cref="NotSupportedException">No registered importer can open one of the paths.</exception>
    public ModelSourceSet OpenSet(string path, IEnumerable<KeyValuePair<string, string>>? auxiliary = null)
    {
        var primary = Open(path);
        var opened = new Dictionary<string, ModelSource>();
        try
        {
            foreach (var (role, auxiliaryPath) in auxiliary ?? [])
            {
                opened[role] = Open(auxiliaryPath);
            }
        }
        catch
        {
            primary.DisposeResources();
            foreach (var source in opened.Values)
            {
                source.DisposeResources();
            }

            throw;
        }

        return new ModelSourceSet { Primary = primary, Auxiliary = opened };
    }
}
