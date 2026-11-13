using System.Buffers;
using System.Net;

namespace Sharplight.Hub;

/// <summary>
/// One download into a <c>.partial</c> file: resumes what is on disk, retries transient failures, hashes while writing and
/// verifies the result. <see cref="FileDownloader"/> renames the file once <see cref="RunAsync"/> returned.
/// </summary>
internal sealed class FileTransfer : IAsyncDisposable
{
    private const int BufferSize = 1 << 20;

    private readonly HttpClient _http;
    private readonly HubClientOptions _options;
    private readonly DownloadRequest _request;
    private readonly IProgress<DownloadProgress>? _progress;
    private readonly Action<string>? _log;
    private readonly string _name;
    private readonly FileStream _file;
    private readonly ContentHasher _hasher;
    private long _received;
    private long _resumed;

    public FileTransfer(HttpClient http, HubClientOptions options, DownloadRequest request, IProgress<DownloadProgress>? progress, Action<string>? log)
    {
        _http = http;
        _options = options;
        _request = request;
        _progress = progress;
        _log = log;
        _name = Path.GetFileName(request.DestinationPath);
        _file = new FileStream(request.PartialPath, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.Read, BufferSize, useAsync: true);
        _hasher = ContentHasher.For(request);
    }

    public async Task RunAsync(CancellationToken cancellationToken)
    {
        await ResumeAsync(cancellationToken).ConfigureAwait(false);
        int failures = 0;
        while (_received < _request.Size)
        {
            long before = _received;
            try
            {
                await TransferAsync(cancellationToken).ConfigureAwait(false);
                if (_received < _request.Size)
                {
                    throw new IOException($"the connection closed after {_received} of {_request.Size} bytes");
                }
            }
            catch (Exception error) when (RetryPolicy.IsTransient(error, cancellationToken))
            {
                // Progress means the connection works: count failures from scratch.
                failures = _received > before ? 1 : failures + 1;
                if (failures >= _options.Retry.MaxAttempts)
                {
                    throw new HubException($"Download of {_name} failed after {failures} attempts without progress: {RetryPolicy.Describe(error)}. "
                        + "Run the command again to resume.", inner: error);
                }

                var delay = RetryPolicy.Delay(_options.Retry, failures, error);
                _log?.Invoke($"{_name}: {RetryPolicy.Describe(error)}; retrying in {delay.TotalSeconds:F0} s (attempt {failures + 1} of {_options.Retry.MaxAttempts})");
                await Task.Delay(delay, cancellationToken).ConfigureAwait(false);
            }
        }

        await _file.FlushAsync(cancellationToken).ConfigureAwait(false);
        await _file.DisposeAsync().ConfigureAwait(false);
        Verify();
    }

    public async ValueTask DisposeAsync()
    {
        await _file.DisposeAsync().ConfigureAwait(false);
        _hasher.Dispose();
    }

    /// <summary>Hashes the bytes an earlier attempt left in the <c>.partial</c> file so the download continues after them.</summary>
    private async Task ResumeAsync(CancellationToken cancellationToken)
    {
        if (_file.Length > _request.Size)
        {
            _file.SetLength(0);
        }

        var buffer = ArrayPool<byte>.Shared.Rent(BufferSize);
        try
        {
            _file.Position = 0;
            int read;
            while ((read = await _file.ReadAsync(buffer.AsMemory(0, BufferSize), cancellationToken).ConfigureAwait(false)) > 0)
            {
                _hasher.Append(buffer.AsSpan(0, read));
                _received += read;
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }

        _resumed = _received;
        Report();
    }

    /// <summary>One HTTP request from the current offset to the end (or until the connection fails).</summary>
    private async Task TransferAsync(CancellationToken cancellationToken)
    {
        using var stall = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        stall.CancelAfter(_options.StallTimeout);
        using var request = HubRequests.Get(_request.Uri, _options);
        if (_received > 0)
        {
            request.Headers.Range = new System.Net.Http.Headers.RangeHeaderValue(_received, null);
        }

        using var response = await _http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, stall.Token).ConfigureAwait(false);
        if (!AcceptResponse(response))
        {
            return;
        }

        var stream = await response.Content.ReadAsStreamAsync(stall.Token).ConfigureAwait(false);
        await using (stream.ConfigureAwait(false))
        {
            await CopyAsync(stream, stall, cancellationToken).ConfigureAwait(false);
        }
    }

    /// <summary>Checks the status and the returned range; returns false when there is nothing to read.</summary>
    private bool AcceptResponse(HttpResponseMessage response)
    {
        switch (response.StatusCode)
        {
            case HttpStatusCode.PartialContent:
                if (response.Content.Headers.ContentRange?.From != _received)
                {
                    Restart();
                    throw new IOException("the server returned a different byte range than requested");
                }

                return true;
            case HttpStatusCode.OK:
                // The server ignored the range: the body starts at the first byte.
                Restart();
                return true;
            case HttpStatusCode.RequestedRangeNotSatisfiable when _received == _request.Size:
                return false;
            case HttpStatusCode.RequestedRangeNotSatisfiable:
                Restart();
                throw new IOException("the server rejected the resume offset");
            default:
                throw HubRequests.Failure(response, _name, RepositoryOf(_request.Uri), _options);
        }
    }

    private async Task CopyAsync(Stream stream, CancellationTokenSource stall, CancellationToken cancellationToken)
    {
        var buffer = ArrayPool<byte>.Shared.Rent(BufferSize);
        try
        {
            _file.Position = _received;
            while (true)
            {
                stall.CancelAfter(_options.StallTimeout);
                int read = await stream.ReadAsync(buffer.AsMemory(0, BufferSize), stall.Token).ConfigureAwait(false);
                if (read == 0)
                {
                    break;
                }

                if (_received + read > _request.Size)
                {
                    Restart();
                    throw new HubException($"{_name}: the server sent more than the expected {_request.Size} bytes; the file changed on the Hub. Run the command again.");
                }

                await _file.WriteAsync(buffer.AsMemory(0, read), cancellationToken).ConfigureAwait(false);
                _hasher.Append(buffer.AsSpan(0, read));
                _received += read;
                Report();
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }
    }

    private void Restart()
    {
        _file.SetLength(0);
        _hasher.Reset();
        _received = 0;
        _resumed = 0;
    }

    private void Report()
    {
        _progress?.Report(new DownloadProgress(_name, _received, _request.Size, _received, _request.Size, _resumed));
    }

    private void Verify()
    {
        var actual = _hasher.Finish();
        if (actual is null || actual == _hasher.Expected)
        {
            return;
        }

        File.Delete(_request.PartialPath);
        throw new HubException($"{_name}: checksum mismatch (expected {_hasher.Expected}, got {actual}). The damaged download was deleted; run the command again.");
    }

    /// <summary>The <c>owner/name</c> part of a resolve address, for error messages.</summary>
    private static string RepositoryOf(Uri uri)
    {
        var segments = uri.AbsolutePath.Trim('/').Split('/');
        return segments.Length >= 2 ? $"{segments[0]}/{segments[1]}" : uri.Host;
    }
}
