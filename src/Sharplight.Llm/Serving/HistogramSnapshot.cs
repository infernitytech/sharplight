namespace Sharplight.Llm.Serving;

/// <summary>Immutable histogram snapshot: cumulative-free per-bucket counts (last entry = overflow), sum and count.</summary>
/// <param name="UpperBounds">Inclusive upper bounds of the buckets in ascending order (seconds for latencies, rows for counts).</param>
/// <param name="Counts">Observations per bucket; one more entry than <paramref name="UpperBounds"/> (the overflow bucket).</param>
/// <param name="Sum">Sum of all observed values.</param>
/// <param name="Count">Number of observations.</param>
public sealed record HistogramSnapshot(double[] UpperBounds, long[] Counts, double Sum, long Count)
{
    /// <summary>Average observed value (0 when empty).</summary>
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

            acc += Counts[i];
            lower = upper;
        }
        return lower;
    }
}
