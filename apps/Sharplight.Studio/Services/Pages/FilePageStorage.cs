namespace Sharplight.Studio.Services.Pages;

/// <summary>A <see cref="IPageStorage"/> with one directory per page under a root (the data directory of the app).</summary>
/// <param name="root">The directory that holds the page directories.</param>
/// <param name="generations">The directory that holds the generated media (<c>images</c> ...); next to <paramref name="root"/> when null.</param>
internal sealed class FilePageStorage(string root, string? generations = null) : IPageStorage
{
    /// <inheritdoc/>
    public string DirectoryFor(string pageId)
    {
        string directory = Path.Combine(root, pageId);
        Directory.CreateDirectory(directory);
        return directory;
    }

    /// <inheritdoc/>
    public string GenerationsDirectory(string kind)
    {
        string directory = Path.Combine(generations ?? Path.Combine(Path.GetDirectoryName(Path.TrimEndingDirectorySeparator(root)) ?? root, "generations"), kind);
        Directory.CreateDirectory(directory);
        return directory;
    }
}
