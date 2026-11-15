namespace Sharplight.Llm.Serving;

/// <summary>Point-in-time view of the engine's counters, gauges and histograms.</summary>
public sealed record EngineMetricsSnapshot
{
    /// <summary>Requests waiting for a slot.</summary>
    public int QueueDepth { get; init; }
    /// <summary>Requests currently holding a slot.</summary>
    public int ActiveSequences { get; init; }
    /// <summary>Total slots of the decoder.</summary>
    public int SlotCount { get; init; }
    /// <summary>Fraction of slots in use (0..1).</summary>
    public double SlotUtilization => SlotCount == 0 ? 0 : (double)ActiveSequences / SlotCount;
    /// <summary>Requests accepted since start.</summary>
    public long RequestsSubmitted { get; init; }
    /// <summary>Requests refused at admission (queue full, prompt too long, draining, invalid).</summary>
    public long RequestsRejected { get; init; }
    /// <summary>Completed requests by finish reason.</summary>
    public IReadOnlyDictionary<FinishReason, long> RequestsFinished { get; init; } = new Dictionary<FinishReason, long>();
    /// <summary>Prompt tokens of admitted requests.</summary>
    public long PromptTokens { get; init; }
    /// <summary>Prompt tokens served from the prefix cache (not recomputed).</summary>
    public long CachedPromptTokens { get; init; }
    /// <summary>Tokens generated.</summary>
    public long CompletionTokens { get; init; }
    /// <summary>Admissions that looked for a reusable prefix.</summary>
    public long PrefixCacheLookups { get; init; }
    /// <summary>Admissions that reused at least one cached token.</summary>
    public long PrefixCacheHits { get; init; }
    /// <summary>Hits divided by lookups (0 without lookups).</summary>
    public double PrefixCacheHitRate => PrefixCacheLookups == 0 ? 0 : (double)PrefixCacheHits / PrefixCacheLookups;
    /// <summary>Forward passes executed.</summary>
    public long Steps { get; init; }
    /// <summary>Completion tokens per second over the last ~10 s.</summary>
    public double RecentTokensPerSecond { get; init; }
    /// <summary>KV-cache capacity in tokens: slots times the context length of a slot.</summary>
    public long KvCapacityTokens { get; init; }
    /// <summary>Tokens held in slot contexts, including idle slots kept for prefix reuse.</summary>
    public long KvTokens { get; init; }
    /// <summary>Tokens held by running sequences.</summary>
    public long KvActiveTokens { get; init; }
    /// <summary>Fraction of the KV capacity held by running sequences (0..1).</summary>
    public double KvUtilization => KvCapacityTokens == 0 ? 0 : (double)KvActiveTokens / KvCapacityTokens;
    /// <summary>Draft tokens proposed by speculative decoding and verified by the model.</summary>
    public long DraftTokens { get; init; }
    /// <summary>Draft tokens the model accepted.</summary>
    public long AcceptedDraftTokens { get; init; }
    /// <summary>Running sequences whose KV blocks were taken back for others (paged cache) and recomputed later.</summary>
    public long Preemptions { get; init; }
    /// <summary>Prompt tokens served from KV blocks shared with another slot's context (paged cache).</summary>
    public long SharedPrefixTokens { get; init; }
    /// <summary>Accepted divided by proposed draft tokens (0 without speculation).</summary>
    public double DraftAcceptanceRate => DraftTokens == 0 ? 0 : (double)AcceptedDraftTokens / DraftTokens;
    /// <summary>Seconds from submission to the first generated token.</summary>
    public HistogramSnapshot TimeToFirstToken { get; init; } = null!;
    /// <summary>Seconds between consecutive tokens of a sequence.</summary>
    public HistogramSnapshot InterTokenLatency { get; init; } = null!;
    /// <summary>Seconds a request waited for a slot.</summary>
    public HistogramSnapshot QueueWait { get; init; } = null!;
    /// <summary>Seconds from submission to the end of a request.</summary>
    public HistogramSnapshot RequestDuration { get; init; } = null!;
    /// <summary>Seconds per forward pass.</summary>
    public HistogramSnapshot StepDuration { get; init; } = null!;
    /// <summary>Rows (tokens) per forward pass.</summary>
    public HistogramSnapshot StepRows { get; init; } = null!;
    /// <summary>Sequences per forward pass (the batch size).</summary>
    public HistogramSnapshot StepSequences { get; init; } = null!;
}
