namespace Sharplight.Hub;

/// <summary>Turns the per-file progress of a sequence of downloads into progress over all of them.</summary>
internal sealed class AggregateProgress(IProgress<DownloadProgress>? target, long total) : IProgress<DownloadProgress>
{
    private long _completed;
    private long _resumed;
    private long _transferred;
    private DownloadProgress _current;

    /// <summary>Bytes transferred over the network so far (excluding bytes that were already on disk).</summary>
    public long Transferred => _transferred + Math.Max(0, _current.FileReceived - _current.Resumed);

    public void Report(DownloadProgress value)
    {
        _current = value;
        target?.Report(new DownloadProgress(value.File, value.FileReceived, value.FileSize, _completed + value.FileReceived, total, _resumed + value.Resumed));
    }

    /// <summary>Moves on to the next file after the current one (of <paramref name="size"/> bytes) completed.</summary>
    public void CompleteFile(long size)
    {
        _completed += size;
        _resumed += _current.Resumed;
        _transferred += Math.Max(0, _current.FileReceived - _current.Resumed);
        _current = default;
    }
}
