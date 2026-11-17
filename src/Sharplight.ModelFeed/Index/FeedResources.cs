namespace Sharplight.ModelFeed.Index;

/// <summary>Relative paths of a feed's documents.</summary>
public sealed record FeedResources
{
    /// <summary>Catalog index.</summary>
    public string Catalog { get; init; } = FeedLayout.CatalogIndex;

    /// <summary>Package index template with <c>{id}</c>.</summary>
    public string Packages { get; init; } = FeedLayout.PackageIndexTemplate;

    /// <summary>Key set.</summary>
    public string Keys { get; init; } = FeedLayout.KeySet;
}
