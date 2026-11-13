namespace Sharplight.Llm.Serving;

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
