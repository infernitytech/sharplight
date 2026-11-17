namespace Sharplight.Hub.Catalog;

/// <summary>The organization or person that made a catalog model (the feed's <c>publisher</c>; admins filter and set policy by country of origin).</summary>
public sealed record CatalogPublisher
{
    /// <summary>Publisher name (<c>Alibaba Cloud (Qwen)</c>).</summary>
    public required string Name { get; init; }

    /// <summary>ISO 3166-1 alpha-2 country code of the publisher's seat (<c>CN</c>, <c>US</c>), or <see langword="null"/> when unknown.</summary>
    public string? CountryOfOrigin { get; init; }

    /// <summary>Publisher web page, or <see langword="null"/>.</summary>
    public string? Url { get; init; }
}
