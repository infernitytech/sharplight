using System.Diagnostics;

namespace Sharplight.Llm.Serving;

/// <summary>Immutable histogram snapshot: cumulative-free per-bucket counts (last entry = overflow), sum and count.</summary>
public sealed record HistogramSnapshot(double[] UpperBounds, long[] Counts, double Sum, long Count)
{
    public double Mean => Count == 0 ? 0 : Sum / Count;
    /// <summary>Estimated quantile (linear interpolation inside the bucket).</summary>
    public double Quantile(double q)
    {
        if (Count == 0)
        {
            return 0;
        }

        double target = q * Count, acc = 0, lower = 0;
        for (int i = 0; i < Counts.Length; i++)
        {
            double upper = i < UpperBounds.Length ? UpperBounds[i] : UpperBounds[^1] * 2;
            if (acc + Counts[i] >= target && Counts[i] > 0)
            {
                return lower + (upper - lower) * (target - acc) / Counts[i];
            }

            acc += Counts[i]; lower = upper;
        }
        return lower;
    }
}

/// <summary>Thread-safe fixed-bucket histogram.</summary>
public sealed class Histogram(double[] upperBounds)
{
    private readonly long[] _counts = new long[upperBounds.Length + 1];
    private double _sum; private long _count;
    private readonly Lock _lock = new();

    public static double[] LatencyBuckets { get; } = [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120];
    public static double[] CountBuckets { get; } = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];

    public void Observe(double v)
    {
        int i = Array.FindIndex(upperBounds, b => v <= b);
        if (i < 0)
        {
            i = upperBounds.Length;
        }

        lock (_lock) { _counts[i]++; _sum += v; _count++; }
    }

    public HistogramSnapshot Snapshot()
    {
        lock (_lock)
        {
            return new HistogramSnapshot(upperBounds, (long[])_counts.Clone(), _sum, _count);
        }
    }
}

/// <summary>Point-in-time view of the engine's counters, gauges and histograms.</summary>
public sealed record EngineMetricsSnapshot
{
    public int QueueDepth { get; init; }
    public int ActiveSequences { get; init; }
    public int SlotCount { get; init; }
    public double SlotUtilization => SlotCount == 0 ? 0 : (double)ActiveSequences / SlotCount;
    public long RequestsSubmitted { get; init; }
    public long RequestsRejected { get; init; }
    public IReadOnlyDictionary<FinishReason, long> RequestsFinished { get; init; } = new Dictionary<FinishReason, long>();
    public long PromptTokens { get; init; }
    /// <summary>Prompt tokens served from the prefix cache (not recomputed).</summary>
    public long CachedPromptTokens { get; init; }
    public long CompletionTokens { get; init; }
    public long PrefixCacheLookups { get; init; }
    public long PrefixCacheHits { get; init; }
    public double PrefixCacheHitRate => PrefixCacheLookups == 0 ? 0 : (double)PrefixCacheHits / PrefixCacheLookups;
    public long Steps { get; init; }
    /// <summary>Completion tokens per second over the last ~10 s.</summary>
    public double RecentTokensPerSecond { get; init; }
    public HistogramSnapshot TimeToFirstToken { get; init; } = null!;
    public HistogramSnapshot InterTokenLatency { get; init; } = null!;
    public HistogramSnapshot QueueWait { get; init; } = null!;
    public HistogramSnapshot RequestDuration { get; init; } = null!;
    public HistogramSnapshot StepDuration { get; init; } = null!;
    public HistogramSnapshot StepRows { get; init; } = null!;
}

/// <summary>Counters and histograms the engine updates; read them as an <see cref="EngineMetricsSnapshot"/>.</summary>
public sealed class EngineMetrics
{
    private long _submitted, _rejected, _promptTokens, _cachedTokens, _completionTokens, _lookups, _hits, _steps;
    private readonly long[] _finished = new long[Enum.GetValues<FinishReason>().Length];
    private readonly Histogram _ttft = new(Histogram.LatencyBuckets), _itl = new(Histogram.LatencyBuckets), _wait = new(Histogram.LatencyBuckets),
        _duration = new(Histogram.LatencyBuckets), _stepTime = new(Histogram.LatencyBuckets), _rows = new(Histogram.CountBuckets);
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

            while (_ringSecond < sec) { _ringSecond++; _ring[_ringSecond % _ring.Length] = 0; }
            _ring[sec % _ring.Length]++;
        }
    }
    internal void Finished(FinishReason r, TimeSpan duration)
    {
        Interlocked.Increment(ref _finished[(int)r]);
        _duration.Observe(duration.TotalSeconds);
    }
    internal void Step(int rows, TimeSpan time)
    {
        Interlocked.Increment(ref _steps);
        _rows.Observe(rows);
        _stepTime.Observe(time.TotalSeconds);
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

    public EngineMetricsSnapshot Snapshot(int queueDepth, int active, int slots) => new()
    {
        QueueDepth = queueDepth, ActiveSequences = active, SlotCount = slots,
        RequestsSubmitted = Interlocked.Read(ref _submitted), RequestsRejected = Interlocked.Read(ref _rejected),
        RequestsFinished = Enum.GetValues<FinishReason>().ToDictionary(r => r, r => Interlocked.Read(ref _finished[(int)r])),
        PromptTokens = Interlocked.Read(ref _promptTokens), CachedPromptTokens = Interlocked.Read(ref _cachedTokens),
        CompletionTokens = Interlocked.Read(ref _completionTokens),
        PrefixCacheLookups = Interlocked.Read(ref _lookups), PrefixCacheHits = Interlocked.Read(ref _hits),
        Steps = Interlocked.Read(ref _steps), RecentTokensPerSecond = RecentRate(),
        TimeToFirstToken = _ttft.Snapshot(), InterTokenLatency = _itl.Snapshot(), QueueWait = _wait.Snapshot(),
        RequestDuration = _duration.Snapshot(), StepDuration = _stepTime.Snapshot(), StepRows = _rows.Snapshot(),
    };
}
