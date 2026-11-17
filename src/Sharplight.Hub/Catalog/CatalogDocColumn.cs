namespace Sharplight.Hub.Catalog;

/// <summary>A column of a generated documentation table.</summary>
public sealed record CatalogDocColumn
{
    /// <summary>
    /// The cell source: <c>model</c>, <c>repositories</c>, <c>formats</c>, <c>providers</c>, <c>verification</c>, <c>notes</c>,
    /// <c>date</c>, <c>kind</c>, or any key that only rows give (<see cref="CatalogDocRow.Cells"/>).
    /// </summary>
    public required string Key { get; init; }

    /// <summary>The header text.</summary>
    public required string Title { get; init; }
}
