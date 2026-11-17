using System.Text.Json.Serialization;

namespace Sharplight.Hub.Catalog;

/// <summary>
/// The verified-model catalog (<c>catalog/verified-models.json</c>, docs/verified-catalog.md): models checked from download to
/// execution, each with pinned revisions and exact file lists, so they can be installed knowing that they work.
/// </summary>
public sealed record VerifiedCatalog
{
    /// <summary>The schema version this library reads and writes.</summary>
    public const int CurrentSchemaVersion = 1;

    /// <summary>The JSON schema the file names for editors (<c>$schema</c>), or <see langword="null"/>.</summary>
    [JsonPropertyName("$schema")]
    public string? Schema { get; init; }

    /// <summary>The schema version of the file.</summary>
    public int SchemaVersion { get; init; } = CurrentSchemaVersion;

    /// <summary>When the catalog was last changed, <c>yyyy-MM-dd</c>; a newer copy replaces an older one.</summary>
    public string Updated { get; init; } = "";

    /// <summary>The generated tables of docs/compatible-models.md.</summary>
    public IReadOnlyList<CatalogDocSection> DocSections { get; init; } = [];

    /// <summary>The models.</summary>
    public IReadOnlyList<CatalogEntry> Entries { get; init; } = [];

    /// <summary>The entry with <paramref name="id"/>, or <see langword="null"/>.</summary>
    /// <param name="id">The entry id (case-insensitive).</param>
    public CatalogEntry? Find(string id) => Entries.FirstOrDefault(entry => string.Equals(entry.Id, id, StringComparison.OrdinalIgnoreCase));
}
