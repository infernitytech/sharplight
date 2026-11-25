namespace Sharplight.ModelFeed.Index;

/// <summary>Paths of a feed's documents, relative to the feed root (channel documents include the channel folder).</summary>
public sealed record FeedResources
{
    /// <summary>Catalog index.</summary>
    public string Catalog { get; init; } = FeedLayout.CatalogIndex;

    /// <summary>The model listing (<see cref="FeedModelListing"/>), or <see langword="null"/> for a feed published without one.</summary>
    public string? Models { get; init; }

    /// <summary>Package index template with <c>{id}</c>.</summary>
    public string Packages { get; init; } = FeedLayout.PackageIndexTemplate;

    /// <summary>Key set.</summary>
    public string Keys { get; init; } = FeedLayout.KeySet;
}
