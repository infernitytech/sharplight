namespace Sharplight.ModelFeed.Index;

/// <summary>Summary of the catalog in the service index.</summary>
public sealed record FeedCatalogSummary
{
    /// <summary>Number of pages.</summary>
    public int Pages { get; init; }

    /// <summary>Number of packages.</summary>
    public int Packages { get; init; }

    /// <summary>Digest of the catalog index (<c>sha256:&lt;hex&gt;</c>).</summary>
    public string Digest { get; init; } = "";
}
