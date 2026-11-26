namespace Sharplight.Hub;

/// <summary>A file a pull downloads: its repository (for the pinned address), the file and where it goes.</summary>
/// <param name="Repository">The resolved repository the file belongs to.</param>
/// <param name="File">The file.</param>
/// <param name="LocalPath">The destination path.</param>
/// <param name="DirectUri">The download address of a file that is not on the Hub (a catalog set from GitHub), or <see langword="null"/>.</param>
internal sealed record PlannedFile(HubRepository Repository, HubFile File, string LocalPath, Uri? DirectUri = null);
