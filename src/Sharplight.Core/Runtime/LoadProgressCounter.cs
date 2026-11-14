namespace Sharplight.Runtime;

/// <summary>
/// Thread-safe counter of one load stage that reports to an <see cref="IProgress{T}"/> at a bounded rate: the first report
/// (0 of the total) when created, then whenever at least a <c>1/maxReports</c> share of the total has completed since the last
/// report, and always when the total is reached. Reports are monotonic and never exceed the total. Without a progress receiver
/// it only counts, so providers can use it unconditionally.
/// </summary>
public sealed class LoadProgressCounter
{
    private readonly IProgress<LoadProgress>? _progress;
    private readonly LoadStage _stage;
    private readonly string? _detail;
    private readonly long _cached;
    private readonly long _step;
    private readonly Lock _gate = new();
    private long _completed;
    private long _reported = -1;

    /// <summary>Creates the counter and reports 0 of <paramref name="total"/>.</summary>
    /// <param name="progress">Receiver of the reports, or <see langword="null"/> to count silently.</param>
    /// <param name="stage">The stage counted.</param>
    /// <param name="total">Units of work of the stage.</param>
    /// <param name="detail">Detail text of every report.</param>
    /// <param name="cached">Value of <see cref="LoadProgress.Cached"/> in every report.</param>
    /// <param name="maxReports">Upper bound of reports between the first and the last.</param>
    public LoadProgressCounter(IProgress<LoadProgress>? progress, LoadStage stage, long total, string? detail = null, long cached = 0, int maxReports = 100)
    {
        _progress = progress;
        _stage = stage;
        _detail = detail;
        _cached = cached;
        Total = Math.Max(0, total);
        _step = Math.Max(1, Total / Math.Max(1, maxReports));
        lock (_gate)
        {
            Report(0, force: true);
        }
    }

    /// <summary>Units of work of the stage.</summary>
    public long Total { get; }

    /// <summary>Units completed so far.</summary>
    public long Completed => Interlocked.Read(ref _completed);

    /// <summary>Adds completed units, reporting when enough have accumulated or the total is reached.</summary>
    /// <param name="units">Units just completed.</param>
    public void Add(long units = 1)
    {
        if (_progress is null)
        {
            Interlocked.Add(ref _completed, units);
            return;
        }

        lock (_gate)
        {
            _completed += units;
            Report(_completed, force: false);
        }
    }

    /// <summary>Marks the stage as done, reporting the total unless it was reported already.</summary>
    public void Complete()
    {
        lock (_gate)
        {
            _completed = Math.Max(_completed, Total);
            Report(_completed, force: false);
        }
    }

    private void Report(long completed, bool force)
    {
        long value = Math.Min(completed, Total);
        if (_progress is null || value <= _reported || (!force && value < Total && value - _reported < _step))
        {
            return;
        }

        _reported = value;
        _progress.Report(new LoadProgress(_stage, value, Total, _detail) { Cached = _cached });
    }
}
