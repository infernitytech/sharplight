namespace Sharplight.ModelFeed.Publishing;

/// <summary>Options of <see cref="FeedPublisher"/>.</summary>
public sealed record FeedPublishOptions
{
    /// <summary>Feed name in the service index.</summary>
    public string Name { get; init; } = "Sharplight model feed";

    /// <summary>How long a published index stays valid (freeze protection: republish before it expires).</summary>
    public TimeSpan Validity { get; init; } = TimeSpan.FromDays(30);

    /// <summary>Package summaries per catalog page.</summary>
    public int PageSize { get; init; } = 200;
}
