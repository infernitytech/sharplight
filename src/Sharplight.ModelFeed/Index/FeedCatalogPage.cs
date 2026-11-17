namespace Sharplight.ModelFeed.Index;

/// <summary>A page of package summaries (<c>catalog/page/&lt;n&gt;.json</c>), ordered by id.</summary>
public sealed record FeedCatalogPage
{
    /// <summary>Schema version (1).</summary>
    public int SchemaVersion { get; init; } = 1;

    /// <summary>Summaries.</summary>
    public IReadOnlyList<FeedPackageSummary> Packages { get; init; } = [];
}
