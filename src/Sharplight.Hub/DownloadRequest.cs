namespace Sharplight.Hub;

/// <summary>One file to download with <see cref="FileDownloader"/>.</summary>
/// <param name="Uri">Where to download from (redirects are followed).</param>
/// <param name="DestinationPath">The local file; written as <c>DestinationPath.partial</c> and renamed once complete and verified.</param>
/// <param name="Size">Expected size in bytes.</param>
/// <param name="Sha256">Expected lower-case hex SHA-256 of the content, or <see langword="null"/>.</param>
/// <param name="GitOid">Expected git blob id (used when <paramref name="Sha256"/> is null), or <see langword="null"/> for no check.</param>
public sealed record DownloadRequest(Uri Uri, string DestinationPath, long Size, string? Sha256 = null, string? GitOid = null)
{
    /// <summary>Extension of the file a download is written to until it is complete and verified.</summary>
    public const string PartialExtension = ".partial";

    /// <summary>The file the download is written to before it is verified.</summary>
    public string PartialPath => DestinationPath + PartialExtension;

    /// <summary>A request for <paramref name="file"/> of a resolved repository.</summary>
    /// <param name="hub">The client that resolved the repository.</param>
    /// <param name="repository">The repository, pinned to its commit.</param>
    /// <param name="file">The file to download.</param>
    /// <param name="destinationPath">The local file.</param>
    public static DownloadRequest For(HubClient hub, HubRepository repository, HubFile file, string destinationPath)
    {
        return new DownloadRequest(hub.GetFileUri(repository, file.Path), destinationPath, file.Size, file.Sha256, file.GitOid);
    }
}
