namespace Sharplight.Runtime;

/// <summary>
/// Combines the progress reports of the segments of one compilation into one count per stage: each segment's completed and
/// total work is added to that of the segments compiled before it (see <see cref="NextSegment"/>), so a receiver sees one
/// growing <c>Completed / Total</c> per stage instead of a count restarting with every segment. Thread-safe.
/// </summary>
internal sealed class SegmentProgress(IProgress<LoadProgress> target) : IProgress<LoadProgress>
{
    private readonly Lock _gate = new();
    private readonly Dictionary<LoadStage, LoadProgress> _done = [];
    private readonly Dictionary<LoadStage, LoadProgress> _current = [];

    /// <inheritdoc/>
    public void Report(LoadProgress value)
    {
        // Forwarded under the lock so that reports of concurrent stages and threads stay in order.
        lock (_gate)
        {
            _current[value.Stage] = value;
            target.Report(_done.TryGetValue(value.Stage, out var d)
                ? value with { Completed = d.Completed + value.Completed, Total = d.Total + value.Total, Cached = d.Cached + value.Cached }
                : value);
        }
    }

    /// <summary>Folds the counts of the segment just compiled into the base of the following segments.</summary>
    public void NextSegment()
    {
        lock (_gate)
        {
            foreach (var (stage, p) in _current)
            {
                _done[stage] = _done.TryGetValue(stage, out var d)
                    ? d with { Completed = d.Completed + p.Completed, Total = d.Total + p.Total, Cached = d.Cached + p.Cached }
                    : p;
            }

            _current.Clear();
        }
    }
}
