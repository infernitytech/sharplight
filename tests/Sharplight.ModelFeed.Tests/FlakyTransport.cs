using Sharplight.ModelFeed.Client;

namespace Sharplight.ModelFeed.Tests;

/// <summary>Breaks package streams after a number of bytes (an aborted transfer).</summary>
internal sealed class FlakyTransport(IFeedTransport inner, long failAfter) : IFeedTransport
{
    public string Location => inner.Location;

    public Task<byte[]?> ReadAsync(string path, long maxBytes, CancellationToken cancellationToken = default) => inner.ReadAsync(path, maxBytes, cancellationToken);

    public async Task<Stream> OpenReadAsync(string path, long offset, CancellationToken cancellationToken = default)
    {
        var stream = await inner.OpenReadAsync(path, offset, cancellationToken);
        using (stream)
        {
            var buffer = new byte[failAfter];
            var read = await stream.ReadAtLeastAsync(buffer, buffer.Length, throwOnEndOfStream: false, cancellationToken);
            return new BreakingStream(buffer.AsMemory(0, read));
        }
    }
}
