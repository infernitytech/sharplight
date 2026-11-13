using System.Collections.Concurrent;

namespace Sharplight.Models;

/// <summary>Global importer registry. Hosts register the formats they want to support.</summary>
public static class ModelImporterRegistry
{
    private static readonly ConcurrentDictionary<string, IModelImporter> s_importers = new(StringComparer.OrdinalIgnoreCase);

    public static void Register(IModelImporter importer) => s_importers[importer.Format] = importer;

    public static IEnumerable<IModelImporter> All => s_importers.Values;

    public static IModelImporter? Find(string path) => s_importers.Values.FirstOrDefault(i => i.CanOpen(path));

    public static ModelSource Open(string path)
        => (Find(path) ?? throw new NotSupportedException(
               $"No registered importer can open '{path}'. Registered formats: {(s_importers.IsEmpty ? "(none)" : string.Join(", ", s_importers.Keys))}"))
           .Open(path);

    /// <summary>Opens a primary model plus auxiliary sources (e.g. ("mmproj", path) for a vision projector).</summary>
    public static ModelSourceSet OpenSet(string path, params (string Role, string Path)[] auxiliary)
    {
        var primary = Open(path);
        var aux = new Dictionary<string, ModelSource>();
        try
        {
            foreach (var (role, p) in auxiliary)
            {
                aux[role] = Open(p);
            }
        }
        catch
        {
            foreach (var r in primary.OwnedResources)
            {
                r.Dispose();
            }

            foreach (var s in aux.Values)
            {
                foreach (var r in s.OwnedResources)
                {
                    r.Dispose();
                }
            }

            throw;
        }
        return new ModelSourceSet { Primary = primary, Auxiliary = aux };
    }
}
