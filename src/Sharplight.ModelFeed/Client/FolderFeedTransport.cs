namespace Sharplight.ModelFeed.Client;

/// <summary>A feed in a local folder (same layout as on object storage; air-gapped mirrors). Thread safe.</summary>
public sealed class FolderFeedTransport : IFeedTransport
{
    private readonly string _root;

    /// <summary>Creates the transport.</summary>
    /// <param name="root">The feed folder.</param>
    public FolderFeedTransport(string root)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(root);
        _root = Path.GetFullPath(root);
    }

    /// <inheritdoc />
    public string Location => new Uri(_root.EndsWith(Path.DirectorySeparatorChar) ? _root : _root + Path.DirectorySeparatorChar).AbsoluteUri;

    /// <inheritdoc />
    public async Task<byte[]?> ReadAsync(string path, long maxBytes, CancellationToken cancellationToken = default)
    {
        var full = Resolve(path);
        if (!File.Exists(full))
        {
            return null;
        }

        var length = new FileInfo(full).Length;
        if (length > maxBytes)
        {
            throw new IOException($"{path} is larger than the allowed {maxBytes} bytes.");
        }

        return await File.ReadAllBytesAsync(full, cancellationToken).ConfigureAwait(false);
    }

    /// <inheritdoc />
    public Task<Stream> OpenReadAsync(string path, long offset, CancellationToken cancellationToken = default)
    {
        var full = Resolve(path);
        if (!File.Exists(full))
        {
            throw new FileNotFoundException($"{path} is not in the feed folder.", full);
        }

        var stream = new FileStream(full, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, FileOptions.Asynchronous | FileOptions.SequentialScan);
        stream.Seek(offset, SeekOrigin.Begin);
        return Task.FromResult<Stream>(stream);
    }

    private string Resolve(string path)
    {
        var full = Path.GetFullPath(Path.Combine(_root, path.Replace('/', Path.DirectorySeparatorChar)));
        if (!full.StartsWith(_root + Path.DirectorySeparatorChar, StringComparison.Ordinal))
        {
            throw new FeedFormatException($"'{path}' points outside the feed.");
        }

        return full;
    }
}
