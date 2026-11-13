namespace Sharplight.Hub;

/// <summary>
/// Downloads large files reliably: writes to <c>.partial</c> files, resumes them with HTTP range requests (also across process
/// restarts), retries transient failures and stalled transfers with exponential backoff, verifies the size and checksum
/// (SHA-256 for LFS files, the git blob id otherwise) and only then renames the file into place. Thread safe; one instance can
/// download several files at once as long as their destinations differ.
/// </summary>
public sealed class FileDownloader
{
    private readonly HttpClient _http;
    private readonly HubClientOptions _options;

    /// <summary>Creates a downloader on <paramref name="http"/> (see <see cref="HubClientOptions.CreateHttpClient"/>), which the caller owns.</summary>
    /// <param name="http">The HTTP client to send requests with.</param>
    /// <param name="options">Token, timeouts and retry settings, or <see langword="null"/> for the defaults.</param>
    public FileDownloader(HttpClient http, HubClientOptions? options = null)
    {
        _http = http;
        _options = options ?? new HubClientOptions();
    }

    /// <summary>
    /// Downloads <paramref name="request"/> unless the destination already exists with the expected size. An existing
    /// <c>.partial</c> file is resumed.
    /// </summary>
    /// <param name="request">What to download and where to.</param>
    /// <param name="progress">Receives progress after every block (synchronously, on the downloading thread), or <see langword="null"/>.</param>
    /// <param name="log">Receives retry notices, or <see langword="null"/>.</param>
    /// <param name="cancellationToken">Cancels the download; the <c>.partial</c> file is kept for a later resume.</param>
    /// <exception cref="HubException">
    /// The server refused the request (authentication, not found), the retries were exhausted, or the content does not match
    /// its checksum (the <c>.partial</c> file is deleted then).
    /// </exception>
    public async Task DownloadAsync(DownloadRequest request, IProgress<DownloadProgress>? progress = null, Action<string>? log = null,
        CancellationToken cancellationToken = default)
    {
        var name = Path.GetFileName(request.DestinationPath);
        if (File.Exists(request.DestinationPath) && new FileInfo(request.DestinationPath).Length == request.Size)
        {
            progress?.Report(new DownloadProgress(name, request.Size, request.Size, request.Size, request.Size, request.Size));
            return;
        }

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(request.DestinationPath))!);
        var transfer = new FileTransfer(_http, _options, request, progress, log);
        await using (transfer.ConfigureAwait(false))
        {
            await transfer.RunAsync(cancellationToken).ConfigureAwait(false);
        }

        File.Move(request.PartialPath, request.DestinationPath, overwrite: true);
    }
}
