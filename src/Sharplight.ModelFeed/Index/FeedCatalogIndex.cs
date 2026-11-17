namespace Sharplight.ModelFeed.Index;

/// <summary>The list of catalog pages (<c>catalog/index.json</c>).</summary>
public sealed record FeedCatalogIndex
{
    /// <summary>Schema version (1).</summary>
    public int SchemaVersion { get; init; } = 1;

    /// <summary>Pages in id order.</summary>
    public IReadOnlyList<FeedPageReference> Pages { get; init; } = [];
}
