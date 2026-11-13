namespace Sharplight.Llm.Serving;

/// <summary>Thread-safe fixed-bucket histogram (lock-free observations).</summary>
internal sealed class Histogram(double[] upperBounds)
{
    private readonly long[] _counts = new long[upperBounds.Length + 1];
    private long _sumBits;

    /// <summary>Request-level latencies: queue wait, time to first token, duration (seconds).</summary>
    public static double[] LatencyBuckets { get; } = [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120];
    /// <summary>Per-token and per-step latencies, which are milliseconds for small models (seconds).</summary>
    public static double[] FineLatencyBuckets { get; } = [0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1, 2.5];
    /// <summary>Counts (rows, sequences per step).</summary>
    public static double[] CountBuckets { get; } = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];

    public void Observe(double v)
    {
        int i = 0;
        while (i < upperBounds.Length && v > upperBounds[i])
        {
            i++;
        }

        Interlocked.Increment(ref _counts[i]);
        long seen = Interlocked.Read(ref _sumBits);
        while (true)
        {
            long updated = BitConverter.DoubleToInt64Bits(BitConverter.Int64BitsToDouble(seen) + v);
            long actual = Interlocked.CompareExchange(ref _sumBits, updated, seen);
            if (actual == seen)
            {
                return;
            }

            seen = actual;
        }
    }

    /// <summary>A snapshot; buckets observed concurrently may be off by the observations in flight.</summary>
    public HistogramSnapshot Snapshot()
    {
        var counts = new long[_counts.Length];
        long total = 0;
        for (int i = 0; i < counts.Length; i++)
        {
            counts[i] = Interlocked.Read(ref _counts[i]);
            total += counts[i];
        }

        return new HistogramSnapshot(upperBounds, counts, BitConverter.Int64BitsToDouble(Interlocked.Read(ref _sumBits)), total);
    }
}
