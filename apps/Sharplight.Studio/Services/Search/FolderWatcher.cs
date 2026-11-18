namespace Sharplight.Studio.Services.Search;

/// <summary>Watches a folder and calls back once, after changes have been quiet for a while (editors and copies write in bursts).</summary>
internal sealed class FolderWatcher : IDisposable
{
    private readonly FileSystemWatcher _watcher;
    private readonly Timer _timer;
    private readonly TimeSpan _quiet;

    /// <param name="folder">The folder (with its subfolders).</param>
    /// <param name="quiet">How long nothing may change before <paramref name="changed"/> is called.</param>
    /// <param name="changed">Called on a thread-pool thread.</param>
    public FolderWatcher(string folder, TimeSpan quiet, Action changed)
    {
        _quiet = quiet;
        _timer = new Timer(_ => changed(), null, Timeout.InfiniteTimeSpan, Timeout.InfiniteTimeSpan);
        _watcher = new FileSystemWatcher(folder)
        {
            IncludeSubdirectories = true,
            NotifyFilter = NotifyFilters.FileName | NotifyFilters.DirectoryName | NotifyFilters.LastWrite | NotifyFilters.Size,
        };
        _watcher.Created += (_, _) => Touch();
        _watcher.Changed += (_, _) => Touch();
        _watcher.Deleted += (_, _) => Touch();
        _watcher.Renamed += (_, _) => Touch();
        _watcher.EnableRaisingEvents = true;
    }

    private void Touch() => _timer.Change(_quiet, Timeout.InfiniteTimeSpan);

    /// <inheritdoc/>
    public void Dispose()
    {
        _watcher.Dispose();
        _timer.Dispose();
    }
}
