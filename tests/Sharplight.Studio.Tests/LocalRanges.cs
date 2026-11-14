namespace Sharplight.Studio.Tests;

/// <summary>Range reads of a local file, answered as the Hub answers range requests, counting the requests and bytes.</summary>
/// <param name="path">The file.</param>
internal sealed class LocalRanges(string path)
{
    /// <summary>Bytes returned so far.</summary>
    public long BytesRead { get; private set; }

    /// <summary>Requests that returned bytes.</summary>
    public int Reads { get; private set; }

    /// <summary>Up to <paramref name="count"/> bytes from <paramref name="offset"/> (none past the end).</summary>
    public Task<byte[]> ReadAsync(long offset, int count, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        using var file = File.OpenRead(path);
        if (offset >= file.Length)
        {
            return Task.FromResult(Array.Empty<byte>());
        }

        file.Position = offset;
        var buffer = new byte[(int)Math.Min(count, file.Length - offset)];
        file.ReadExactly(buffer);
        BytesRead += buffer.Length;
        Reads++;
        return Task.FromResult(buffer);
    }
}
