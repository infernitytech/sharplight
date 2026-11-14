using Sharplight.Llm.Serving;
using Sharplight.Studio.Models;

namespace Sharplight.Studio.Services.Runtime;

/// <summary>
/// A model registered with the runtime: usually loaded on its device, possibly loading, or unloaded for idleness or memory (then it
/// reloads on its next request). Chats and API requests share it.
/// </summary>
internal interface ILoadedModel
{
    /// <summary>The id it is known by (in the chat and to API clients).</summary>
    public string Id { get; }

    /// <summary>The file it is loaded from.</summary>
    public string Path { get; }

    /// <summary>The device it runs on.</summary>
    public ComputeDeviceOption Device { get; }

    /// <summary>The settings it is loaded with.</summary>
    public LoadSettings Settings { get; }

    /// <summary>What it can do (vision only when the engine runs its vision tower); refined once it loaded.</summary>
    public ModelCapabilities Capabilities { get; }

    /// <summary>Context length per sequence in tokens.</summary>
    public int ContextLength { get; }

    /// <summary>Life-cycle state.</summary>
    public ResidentState State { get; }

    /// <summary>Error of the last failed load.</summary>
    public string? Error { get; }

    /// <summary>Progress of a load in progress (stage and counts), else null.</summary>
    public LoadStatus? LoadStatus { get; }

    /// <summary>Requests running on it (chat and API).</summary>
    public int ActiveRequests { get; }

    /// <summary>Device memory it holds while loaded, in bytes.</summary>
    public long ResidentBytes { get; }

    /// <summary>Estimated footprint (weights, KV cache, activations) from the last load, in bytes.</summary>
    public long EstimatedBytes { get; }

    /// <summary>Kept loaded: never evicted or unloaded for idleness.</summary>
    public bool IsPinned { get; }

    /// <summary>When it was added.</summary>
    public DateTimeOffset LoadedAt { get; }

    /// <summary>When it last served a request.</summary>
    public DateTimeOffset? LastUsed { get; }

    /// <summary>Streams an answer (loading the model first when it sleeps); cancelling stops generation.</summary>
    public IAsyncEnumerable<ChatStreamUpdate> StreamChatAsync(ChatInput input, CancellationToken cancellationToken);

    /// <summary>Tokens <paramref name="text"/> occupies (an estimate until the model loaded once).</summary>
    public int CountTokens(string text);

    /// <summary>The scheduler's counters while loaded, else null.</summary>
    public EngineMetricsSnapshot? GetMetrics();
}
