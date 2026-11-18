using Sharplight.ModelFeed.Index;

namespace Sharplight.Studio.Services.Feed;

/// <summary>The packages of the model feed's channel, verified, from the network or from the last verified copy.</summary>
internal sealed record FeedListing
{
    /// <summary>The packages (empty when <see cref="Failure"/> is set).</summary>
    public IReadOnlyList<FeedPackageSummary> Packages { get; init; } = [];

    /// <summary>The list comes from the last verified copy on disk because the feed could not be reached.</summary>
    public bool FromCache { get; init; }

    /// <summary>The index and key set were checked against the trusted root key.</summary>
    public bool Verified { get; init; }

    /// <summary>When the feed index was written; null without a listing.</summary>
    public DateTimeOffset? Updated { get; init; }

    /// <summary>Notes from the verification (for example an expired index accepted offline).</summary>
    public IReadOnlyList<string> Warnings { get; init; } = [];

    /// <summary>Why nothing is listed, or null.</summary>
    public FeedFailure? Failure { get; init; }
}
