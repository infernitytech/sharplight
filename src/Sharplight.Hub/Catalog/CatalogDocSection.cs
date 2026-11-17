namespace Sharplight.Hub.Catalog;

/// <summary>A generated table of docs/compatible-models.md, placed where the document has the section's marker.</summary>
public sealed record CatalogDocSection
{
    /// <summary>The section id the marker names (e.g. <c>1.1</c>).</summary>
    public required string Id { get; init; }

    /// <summary>The columns, in order.</summary>
    public required IReadOnlyList<CatalogDocColumn> Columns { get; init; }
}
