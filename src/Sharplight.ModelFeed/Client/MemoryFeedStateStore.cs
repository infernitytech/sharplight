using System.Collections.Concurrent;

namespace Sharplight.ModelFeed.Client;

/// <summary>Feed state for the lifetime of the process. Thread safe.</summary>
public sealed class MemoryFeedStateStore : IFeedStateStore
{
    private readonly ConcurrentDictionary<string, FeedState> _states = new(StringComparer.Ordinal);

    /// <inheritdoc />
    public Task<FeedState?> GetAsync(string feed, CancellationToken cancellationToken = default) =>
        Task.FromResult(_states.TryGetValue(feed, out var state) ? state : null);

    /// <inheritdoc />
    public Task SetAsync(string feed, FeedState state, CancellationToken cancellationToken = default)
    {
        _states[feed] = state;
        return Task.CompletedTask;
    }
}
