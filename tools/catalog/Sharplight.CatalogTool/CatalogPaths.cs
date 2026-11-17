namespace Sharplight.CatalogTool;

/// <summary>Where the catalog and the generated document live in the repository.</summary>
internal static class CatalogPaths
{
    /// <summary>The catalog, relative to the repository root.</summary>
    public const string Catalog = "catalog/verified-models.json";

    /// <summary>The JSON schema's file name, next to the catalog.</summary>
    public const string SchemaFileName = "verified-models.schema.json";

    /// <summary>The document whose tables are generated, relative to the repository root.</summary>
    public const string Document = "docs/compatible-models.md";

    /// <summary>The repository root: the first directory upwards from <paramref name="start"/> that holds <c>Sharplight.slnx</c>.</summary>
    /// <param name="start">Where to start looking.</param>
    /// <exception cref="DirectoryNotFoundException">No such directory.</exception>
    public static string FindRepositoryRoot(string start)
    {
        for (var directory = new DirectoryInfo(start); directory is not null; directory = directory.Parent)
        {
            if (File.Exists(Path.Combine(directory.FullName, "Sharplight.slnx")))
            {
                return directory.FullName;
            }
        }

        throw new DirectoryNotFoundException($"No Sharplight repository above {start}; pass --catalog and --doc.");
    }
}
