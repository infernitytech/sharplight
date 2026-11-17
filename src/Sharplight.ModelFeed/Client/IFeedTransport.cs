namespace Sharplight.ModelFeed.Client;

/// <summary>Reads objects of a feed by relative path (HTTP, a folder, or a peer). Implementations are thread safe.</summary>
public interface IFeedTransport
{
    /// <summary>Where the feed is (for messages and as the state key).</summary>
    public string Location { get; }

    /// <summary>Reads a small document completely.</summary>
    /// <param name="path">Relative path (<c>index.json</c>).</param>
    /// <param name="maxBytes">Largest accepted size.</param>
    /// <param name="cancellationToken">Cancellation.</param>
    /// <returns>The bytes, or <see langword="null"/> when the object does not exist.</returns>
    /// <exception cref="IOException">The object could not be read (network or file error) or is larger than <paramref name="maxBytes"/>.</exception>
    public Task<byte[]?> ReadAsync(string path, long maxBytes, CancellationToken cancellationToken = default);

    /// <summary>Opens an object for reading from <paramref name="offset"/> (a range request over HTTP).</summary>
    /// <param name="path">Relative path.</param>
    /// <param name="offset">First byte.</param>
    /// <param name="cancellationToken">Cancellation.</param>
    /// <returns>A stream positioned at <paramref name="offset"/>; the caller disposes it.</returns>
    /// <exception cref="IOException">The object does not exist or could not be read.</exception>
    public Task<Stream> OpenReadAsync(string path, long offset, CancellationToken cancellationToken = default);
}
