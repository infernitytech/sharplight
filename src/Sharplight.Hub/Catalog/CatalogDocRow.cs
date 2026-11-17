namespace Sharplight.Hub.Catalog;

/// <summary>
/// A row of a generated table of docs/compatible-models.md. Cells not given are filled from the entry (see
/// <see cref="CatalogDocColumn.Key"/>); one entry may have rows in several sections.
/// </summary>
public sealed record CatalogDocRow
{
    /// <summary>The <see cref="CatalogDocSection.Id"/> of the table.</summary>
    public required string Section { get; init; }

    /// <summary>Cell text by column key, overriding the value derived from the entry.</summary>
    public IReadOnlyDictionary<string, string> Cells { get; init; } = new Dictionary<string, string>();
}
