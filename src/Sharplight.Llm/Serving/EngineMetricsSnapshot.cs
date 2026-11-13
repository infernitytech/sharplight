namespace Sharplight.Llm.Serving;

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
