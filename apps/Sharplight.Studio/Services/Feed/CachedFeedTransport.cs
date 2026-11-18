using Sharplight.ModelFeed.Client;

namespace Sharplight.Studio.Services.Feed;

/// <summary>A feed transport that reads only the <see cref="FeedDocumentCache"/> (no network): documents that were never cached do not exist.</summary>
internal sealed class CachedFeedTransport(string location, FeedDocumentCache cache) : IFeedTransport
{
    /// <inheritdoc/>
    public string Location => location;

    /// <inheritdoc/>
    public Task<byte[]?> ReadAsync(string path, long maxBytes, CancellationToken cancellationToken = default)
    {
        var bytes = cache.Read(path);
        if (bytes is not null && bytes.Length > maxBytes)
        {
            throw new IOException($"{path} is larger than the allowed {maxBytes} bytes.");
        }

        return Task.FromResult(bytes);
    }

    /// <inheritdoc/>
    public Task<Stream> OpenReadAsync(string path, long offset, CancellationToken cancellationToken = default)
        => throw new IOException($"{path} is not available offline.");
}
