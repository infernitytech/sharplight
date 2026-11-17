namespace Sharplight.ModelFeed.Index;

/// <summary>A catalog page with its digest.</summary>
public sealed record FeedPageReference
{
    /// <summary>Relative path.</summary>
    public required string Path { get; init; }

    /// <summary>Packages on the page.</summary>
    public int Count { get; init; }

    /// <summary>First package id.</summary>
    public string? FirstId { get; init; }

    /// <summary>Last package id.</summary>
    public string? LastId { get; init; }

    /// <summary>Digest of the page.</summary>
    public required string Digest { get; init; }
}
