using Sharplight.Llm.Serving.Hosting;

namespace Sharplight.Hosting;

/// <summary>State and load of one registered model.</summary>
public sealed record HostedModelStatus
{
    /// <summary>The registered name.</summary>
    public required string Name { get; init; }

    /// <summary>What it does.</summary>
    public required ModelKind Kind { get; init; }

    /// <summary>The task kind of a task model, else null.</summary>
    public string? Task { get; init; }

    /// <summary>Lifecycle state.</summary>
    public required ModelState State { get; init; }

    /// <summary>Never evicted and never unloaded when idle.</summary>
    public bool Pinned { get; init; }

    /// <summary>Device ids of its last load attempt, primary first.</summary>
    public IReadOnlyList<string> Devices { get; init; } = [];

    /// <summary>Device bytes it holds while loaded.</summary>
    public long ResidentBytes { get; init; }

    /// <summary>Estimated footprint from the last load attempt.</summary>
    public long EstimatedBytes { get; init; }

    /// <summary>Requests holding it right now.</summary>
    public int ActiveLeases { get; init; }

    /// <summary>Requests waiting for a slot (chat models).</summary>
    public int QueueDepth { get; init; }

    /// <summary>Requests being generated (chat models).</summary>
    public int ActiveSequences { get; init; }

    /// <summary>Concurrent sequences it can batch (chat models).</summary>
    public int Slots { get; init; }

    /// <summary>Generated tokens per second over the last seconds (chat models).</summary>
    public double TokensPerSecond { get; init; }

    /// <summary>Requests accepted since it loaded (chat models).</summary>
    public long RequestsSubmitted { get; init; }

    /// <summary>Requests refused since it loaded: queue full, prompt too long, draining (chat models).</summary>
    public long RequestsRejected { get; init; }

    /// <summary>Prompt tokens since it loaded (chat models).</summary>
    public long PromptTokens { get; init; }

    /// <summary>Generated tokens since it loaded (chat models).</summary>
    public long CompletionTokens { get; init; }

    /// <summary>When it finished loading.</summary>
    public DateTimeOffset? LoadedAt { get; init; }

    /// <summary>When it was last used.</summary>
    public DateTimeOffset? LastUsed { get; init; }

    /// <summary>Why its last load failed.</summary>
    public string? Error { get; init; }
}
