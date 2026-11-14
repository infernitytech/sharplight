namespace Sharplight.Studio.Models;

/// <summary>Aggregates over a set of <see cref="RequestRecord"/>s.</summary>
internal sealed record StatsSummary
{
    /// <summary>Number of requests.</summary>
    public int Requests { get; init; }

    /// <summary>Prompt tokens summed.</summary>
    public long PromptTokens { get; init; }

    /// <summary>Generated tokens summed.</summary>
    public long CompletionTokens { get; init; }

    /// <summary>Generated tokens divided by decode time, over requests that generated at least two tokens.</summary>
    public double DecodeTokensPerSecond { get; init; }

    /// <summary>Computed prompt tokens divided by prefill time.</summary>
    public double PrefillTokensPerSecond { get; init; }

    /// <summary>Median time to first token in milliseconds.</summary>
    public double MedianTimeToFirstTokenMs { get; init; }

    /// <summary>95th percentile of the time to first token in milliseconds.</summary>
    public double P95TimeToFirstTokenMs { get; init; }

    /// <summary>Accepted divided by proposed draft tokens, or null without speculation.</summary>
    public double? AcceptanceRate { get; init; }

    /// <summary>Computes the aggregates.</summary>
    public static StatsSummary From(IReadOnlyCollection<RequestRecord> records)
    {
        if (records.Count == 0)
        {
            return new StatsSummary();
        }

        var stats = records.Select(record => record.Stats).ToList();
        var decoding = stats.Where(s => s.CompletionTokens > 1 && s.DecodeMs > 0).ToList();
        var prefilling = stats.Where(s => s.PromptTokens > s.CachedTokens && s.PrefillMs > 0).ToList();
        var ttft = stats.Select(s => s.TimeToFirstTokenMs).Order().ToList();
        long drafted = stats.Sum(s => (long)s.DraftTokens);
        return new StatsSummary
        {
            Requests = records.Count,
            PromptTokens = stats.Sum(s => (long)s.PromptTokens),
            CompletionTokens = stats.Sum(s => (long)s.CompletionTokens),
            DecodeTokensPerSecond = Rate(decoding.Sum(s => s.CompletionTokens - 1.0), decoding.Sum(s => s.DecodeMs)),
            PrefillTokensPerSecond = Rate(prefilling.Sum(s => (double)(s.PromptTokens - s.CachedTokens)), prefilling.Sum(s => s.PrefillMs)),
            MedianTimeToFirstTokenMs = Percentile(ttft, 0.5),
            P95TimeToFirstTokenMs = Percentile(ttft, 0.95),
            AcceptanceRate = drafted == 0 ? null : stats.Sum(s => (double)s.AcceptedDraftTokens) / drafted,
        };
    }

    private static double Rate(double tokens, double milliseconds) => milliseconds <= 0 ? 0 : tokens / (milliseconds / 1000);

    private static double Percentile(List<double> sorted, double fraction)
    {
        if (sorted.Count == 0)
        {
            return 0;
        }

        int index = (int)Math.Ceiling(fraction * sorted.Count) - 1;
        return sorted[Math.Clamp(index, 0, sorted.Count - 1)];
    }
}
