namespace Sharplight.ModelFeed.Client;

/// <summary>Remembers the last accepted index per feed, so an older index (rollback) is rejected. Implementations are thread safe.</summary>
public interface IFeedStateStore
{
    /// <summary>The state of a feed, or <see langword="null"/> when it was never synced.</summary>
    /// <param name="feed">Feed location.</param>
    /// <param name="cancellationToken">Cancellation.</param>
    /// <returns>The state.</returns>
    public Task<FeedState?> GetAsync(string feed, CancellationToken cancellationToken = default);

    /// <summary>Stores the state of a feed.</summary>
    /// <param name="feed">Feed location.</param>
    /// <param name="state">The accepted index.</param>
    /// <param name="cancellationToken">Cancellation.</param>
    /// <returns>A task.</returns>
    public Task SetAsync(string feed, FeedState state, CancellationToken cancellationToken = default);
}
