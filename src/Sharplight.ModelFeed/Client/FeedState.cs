namespace Sharplight.ModelFeed.Client;

/// <summary>What a client remembers about a feed between syncs (rollback protection).</summary>
public sealed record FeedState
{
    /// <summary>Highest sequence accepted.</summary>
    public long Sequence { get; init; }

    /// <summary>Digest of the index accepted at that sequence.</summary>
    public required string IndexDigest { get; init; }

    /// <summary>The index's update time.</summary>
    public DateTimeOffset Updated { get; init; }
}
