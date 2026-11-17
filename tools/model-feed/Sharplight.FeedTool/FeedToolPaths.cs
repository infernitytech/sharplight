namespace Sharplight.FeedTool;

/// <summary>Default locations in the repository.</summary>
internal static class FeedToolPaths
{
    /// <summary>The verified catalog, relative to the repository root.</summary>
    public const string Catalog = "catalog/verified-models.json";

    /// <summary>The feed overlay, relative to the repository root.</summary>
    public const string Overlay = "tools/model-feed/feed-overlay.json";

    /// <summary>A file below the repository that contains the working directory.</summary>
    /// <param name="relative">Path relative to the repository root.</param>
    /// <exception cref="DirectoryNotFoundException">The working directory is not inside a Sharplight repository.</exception>
    public static string InRepository(string relative)
    {
        for (var directory = new DirectoryInfo(Environment.CurrentDirectory); directory is not null; directory = directory.Parent)
        {
            if (File.Exists(Path.Combine(directory.FullName, "Sharplight.slnx")))
            {
                return Path.Combine(directory.FullName, relative);
            }
        }

        throw new DirectoryNotFoundException($"No Sharplight repository above {Environment.CurrentDirectory}; pass the path explicitly.");
    }
}
