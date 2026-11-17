namespace Sharplight.ModelFeed.Index;

/// <summary>The service index of a feed (<c>index.json</c>, signed by <c>index.sig</c>): validity, sequence and the digest of the catalog.</summary>
public sealed record FeedServiceIndex
{
    /// <summary>Schema version (1).</summary>
    public int SchemaVersion { get; init; } = 1;

    /// <summary>Feed name.</summary>
    public string? Name { get; init; }

    /// <summary>When the index was written.</summary>
    public DateTimeOffset Updated { get; init; }

    /// <summary>After this time the index is rejected (freeze protection).</summary>
    public DateTimeOffset Expires { get; init; }

    /// <summary>Strictly increasing per publication (rollback protection).</summary>
    public long Sequence { get; init; }

    /// <summary>Relative paths of the other documents.</summary>
    public FeedResources Resources { get; init; } = new();

    /// <summary>Catalog page count and digest.</summary>
    public FeedCatalogSummary Catalog { get; init; } = new();
}
