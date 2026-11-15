namespace Sharplight.Studio.Services.Pages;

/// <summary>A <see cref="IPageStorage"/> with one directory per page under a root (the data directory of the app).</summary>
/// <param name="root">The directory that holds the page directories.</param>
internal sealed class FilePageStorage(string root) : IPageStorage
{
    /// <inheritdoc/>
    public string DirectoryFor(string pageId)
    {
        string directory = Path.Combine(root, pageId);
        Directory.CreateDirectory(directory);
        return directory;
    }
}
