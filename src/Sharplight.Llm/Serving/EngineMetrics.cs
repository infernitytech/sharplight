using System.Diagnostics;

namespace Sharplight.Llm.Serving;

/// <summary>Counters and histograms the engine updates; read them as an <see cref="EngineMetricsSnapshot"/>.</summary>
public sealed class EngineMetrics
{
    private long _submitted, _rejected, _promptTokens, _cachedTokens, _completionTokens, _lookups, _hits, _steps, _drafted, _accepted;
    private readonly long[] _finished = new long[Enum.GetValues<FinishReason>().Length];
    private readonly Histogram _ttft = new(Histogram.LatencyBuckets), _itl = new(Histogram.FineLatencyBuckets), _wait = new(Histogram.LatencyBuckets),
        _duration = new(Histogram.LatencyBuckets), _stepTime = new(Histogram.FineLatencyBuckets), _rows = new(Histogram.CountBuckets),
        _sequences = new(Histogram.CountBuckets);
    // Ring of per-second completion token counts for the recent rate.
    private readonly long[] _ring = new long[11];
    private long _ringSecond = -1;
    private readonly Lock _ringLock = new();

    internal void Submitted() => Interlocked.Increment(ref _submitted);
    internal void Rejected() => Interlocked.Increment(ref _rejected);
    internal void Admitted(int promptTokens, int cached, TimeSpan wait)
    {
        Interlocked.Add(ref _promptTokens, promptTokens);
        Interlocked.Add(ref _cachedTokens, cached);
        Interlocked.Increment(ref _lookups);
        if (cached > 0)
        {
            Interlocked.Increment(ref _hits);
        }

        _wait.Observe(wait.TotalSeconds);
    }
    internal void FirstToken(TimeSpan sinceSubmit) => _ttft.Observe(sinceSubmit.TotalSeconds);
    internal void Token(TimeSpan? sincePrevious)
    {
        Interlocked.Increment(ref _completionTokens);
        if (sincePrevious is { } d)
        {
            _itl.Observe(d.TotalSeconds);
        }

        long sec = Stopwatch.GetTimestamp() / Stopwatch.Frequency;
        lock (_ringLock)
        {
            if (_ringSecond < 0)
            {
                _ringSecond = sec;
            }

            while (_ringSecond < sec)
            {
                _ringSecond++;
                _ring[_ringSecond % _ring.Length] = 0;
            }
            _ring[sec % _ring.Length]++;
        }
    }
    internal void Finished(FinishReason r, TimeSpan duration)
    {
        Interlocked.Increment(ref _finished[(int)r]);
        _duration.Observe(duration.TotalSeconds);
    }
    internal void Step(int rows, int sequences, TimeSpan time)
    {
        Interlocked.Increment(ref _steps);
        _rows.Observe(rows);
        _sequences.Observe(sequences);
        _stepTime.Observe(time.TotalSeconds);
    }
    internal void Speculation(int drafted, int accepted)
    {
        Interlocked.Add(ref _drafted, drafted);
        Interlocked.Add(ref _accepted, accepted);
    }

    private double RecentRate()
    {
        long sec = Stopwatch.GetTimestamp() / Stopwatch.Frequency;
        lock (_ringLock)
        {
            if (_ringSecond < 0)
            {
                return 0;
            }

            long total = 0;
            for (long s = sec - 9; s <= sec; s++)
            {
                if (s <= _ringSecond && s > _ringSecond - _ring.Length)
                {
                    total += _ring[s % _ring.Length];
                }
            }

            return total / 10.0;
        }
    }

    /// <summary>Reads all counters and histograms; the gauges are supplied by the caller (the engine).</summary>
    public EngineMetricsSnapshot Snapshot(int queueDepth, int active, int slots) => new()
    {
        QueueDepth = queueDepth, ActiveSequences = active, SlotCount = slots,
        RequestsSubmitted = Interlocked.Read(ref _submitted), RequestsRejected = Interlocked.Read(ref _rejected),
        RequestsFinished = Enum.GetValues<FinishReason>().ToDictionary(r => r, r => Interlocked.Read(ref _finished[(int)r])),
        PromptTokens = Interlocked.Read(ref _promptTokens), CachedPromptTokens = Interlocked.Read(ref _cachedTokens),
        CompletionTokens = Interlocked.Read(ref _completionTokens),
        PrefixCacheLookups = Interlocked.Read(ref _lookups), PrefixCacheHits = Interlocked.Read(ref _hits),
        Steps = Interlocked.Read(ref _steps), RecentTokensPerSecond = RecentRate(),
        DraftTokens = Interlocked.Read(ref _drafted), AcceptedDraftTokens = Interlocked.Read(ref _accepted),
        TimeToFirstToken = _ttft.Snapshot(), InterTokenLatency = _itl.Snapshot(), QueueWait = _wait.Snapshot(),
        RequestDuration = _duration.Snapshot(), StepDuration = _stepTime.Snapshot(), StepRows = _rows.Snapshot(),
        StepSequences = _sequences.Snapshot(),
    };
}
