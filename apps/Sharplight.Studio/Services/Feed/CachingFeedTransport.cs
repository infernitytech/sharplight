using System.Collections.Concurrent;
using Sharplight.ModelFeed.Client;

namespace Sharplight.Studio.Services.Feed;

/// <summary>
/// A feed transport that remembers the documents it reads and writes them to a <see cref="FeedDocumentCache"/> only when
/// <see cref="Commit"/> is called, after the <see cref="FeedClient"/> verified the whole snapshot: the cache never holds half of a newer
/// feed state, and an offline read goes through the same verification. Package files pass through.
/// </summary>
internal sealed class CachingFeedTransport(IFeedTransport inner, FeedDocumentCache cache) : IFeedTransport
{
    private readonly ConcurrentDictionary<string, byte[]> _read = new(StringComparer.Ordinal);

    /// <inheritdoc/>
    public string Location => inner.Location;

    /// <inheritdoc/>
    public async Task<byte[]?> ReadAsync(string path, long maxBytes, CancellationToken cancellationToken = default)
    {
        var bytes = await inner.ReadAsync(path, maxBytes, cancellationToken).ConfigureAwait(false);
        if (bytes is not null)
        {
            _read[path] = bytes;
        }

        return bytes;
    }

    /// <inheritdoc/>
    public Task<Stream> OpenReadAsync(string path, long offset, CancellationToken cancellationToken = default)
        => inner.OpenReadAsync(path, offset, cancellationToken);

    /// <summary>Writes the documents read so far to the cache.</summary>
    public void Commit()
    {
        foreach (var (path, bytes) in _read)
        {
            try
            {
                cache.Write(path, bytes);
            }
            catch (Exception error) when (error is IOException or UnauthorizedAccessException)
            {
                // The cache is an optimization: a full or read-only disk does not stop the feed.
                return;
            }
        }
    }
}
