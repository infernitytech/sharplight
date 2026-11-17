namespace Sharplight.ModelFeed.Client;

/// <summary>Feed state in a JSON file (one object keyed by feed location), written atomically. Thread safe within one process.</summary>
public sealed class FileFeedStateStore : IFeedStateStore
{
    private readonly string _path;
    private readonly SemaphoreSlim _lock = new(1, 1);

    /// <summary>Creates the store.</summary>
    /// <param name="path">The state file.</param>
    public FileFeedStateStore(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        _path = Path.GetFullPath(path);
    }

    /// <inheritdoc />
    public async Task<FeedState?> GetAsync(string feed, CancellationToken cancellationToken = default)
    {
        await _lock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            return Load().GetValueOrDefault(feed);
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <inheritdoc />
    public async Task SetAsync(string feed, FeedState state, CancellationToken cancellationToken = default)
    {
        await _lock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            var states = Load();
            states[feed] = state;
            Directory.CreateDirectory(Path.GetDirectoryName(_path)!);
            var temporary = _path + ".tmp";
            await File.WriteAllBytesAsync(temporary, FeedJson.Write(states, FeedJsonContext.Default.DictionaryStringFeedState), cancellationToken).ConfigureAwait(false);
            File.Move(temporary, _path, overwrite: true);
        }
        finally
        {
            _lock.Release();
        }
    }

    private Dictionary<string, FeedState> Load()
    {
        if (!File.Exists(_path))
        {
            return new Dictionary<string, FeedState>(StringComparer.Ordinal);
        }

        var states = FeedJson.Read(File.ReadAllBytes(_path), FeedJsonContext.Default.DictionaryStringFeedState, $"Feed state {_path}");
        return new Dictionary<string, FeedState>(states, StringComparer.Ordinal);
    }
}
