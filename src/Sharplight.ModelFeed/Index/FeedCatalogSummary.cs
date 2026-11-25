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

    /// <summary>Number of model groups in the model listing (<see cref="FeedResources.Models"/>).</summary>
    public int Models { get; init; }

    /// <summary>Digest of the model listing (<c>sha256:&lt;hex&gt;</c>), or <see langword="null"/> for a feed published without one.</summary>
    public string? ModelsDigest { get; init; }
}
