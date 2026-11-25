namespace Sharplight.ModelFeed.Publishing;

/// <summary>Options of <see cref="FeedPublisher"/>.</summary>
public sealed record FeedPublishOptions
{
    /// <summary>The channel to publish to (<see cref="FeedChannels.Unstable"/> for new versions), or <see langword="null"/> for a feed without channels.</summary>
    public string? Channel { get; init; }

    /// <summary>Feed name in the service index.</summary>
    public string Name { get; init; } = "Sharplight model feed";

    /// <summary>How long a published index stays valid (freeze protection: republish before it expires).</summary>
    public TimeSpan Validity { get; init; } = TimeSpan.FromDays(30);

    /// <summary>Package summaries per catalog page.</summary>
    public int PageSize { get; init; } = 200;

    /// <summary>
    /// Move package files from the staging folders into the feed folder instead of copying them (instant on the same volume), so a large
    /// package exists once on disk. The staging folder keeps its manifest; a failed publish leaves the file where it was.
    /// </summary>
    public bool MovePackageFiles { get; init; }
}
