namespace Sharplight.Hub.Catalog;

/// <summary>A model of the verified catalog: what it is, how it was verified and the pinned files of each of its variants.</summary>
public sealed record CatalogEntry
{
    /// <summary>Unique id (lower case, digits, <c>.</c> and <c>-</c>).</summary>
    public required string Id { get; init; }

    /// <summary>Display name.</summary>
    public required string Name { get; init; }

    /// <summary>The task kind (<c>Sharplight.Models.ModelKinds</c>, <c>embeddings</c> or <c>reranker</c>).</summary>
    public required string Kind { get; init; }

    /// <summary>The model family (<c>Qwen3</c>, <c>FLUX.1</c>, ...).</summary>
    public string Family { get; init; } = "";

    /// <summary>The architecture id Sharplight maps it with, or <see langword="null"/>.</summary>
    public string? Architecture { get; init; }

    /// <summary>How it was checked.</summary>
    public CatalogLevel Level { get; init; } = CatalogLevel.Verified;

    /// <summary>The license id the repository declares (<c>apache-2.0</c>, <c>other</c>...), or <see langword="null"/>.</summary>
    public string? License { get; init; }

    /// <summary>A repository of the entry is gated: its terms must be accepted on the Hub and a token set.</summary>
    public bool Gated { get; init; }

    /// <summary>Who made the model (not who repackaged it) and the publisher's country of origin; used by the model feed, or <see langword="null"/> when not recorded.</summary>
    public CatalogPublisher? Publisher { get; init; }

    /// <summary>Capability tags (<c>vision</c>, <c>tools</c>, <c>moe</c>, <c>reasoning</c>, ...).</summary>
    public IReadOnlyList<string> Tags { get; init; } = [];

    /// <summary>The downloadable variants; empty for entries that are only documented (not installable).</summary>
    public IReadOnlyList<CatalogVariant> Variants { get; init; } = [];

    /// <summary>File sets variants install besides their own (<see cref="CatalogVariant.Companions"/>).</summary>
    public IReadOnlyList<CatalogCompanion> Companions { get; init; } = [];

    /// <summary>How and when it was checked.</summary>
    public CatalogVerification Verification { get; init; } = new();

    /// <summary>Limitations and remarks, or <see langword="null"/>.</summary>
    public string? Notes { get; init; }

    /// <summary>
    /// What a user must know about the licence before using the model (non-commercial use, a custom agreement, a revenue limit, gated terms), in one or two sentences;
    /// Studio shows it once per model. <see langword="null"/> for permissive licences. Unknown to older readers, which ignore it.
    /// </summary>
    public string? LicenseNotice { get; init; }

    /// <summary>Its rows in the generated tables of docs/compatible-models.md.</summary>
    public IReadOnlyList<CatalogDocRow> Doc { get; init; } = [];

    /// <summary>Whether Studio offers it for download: verified or partial, with at least one variant.</summary>
    public bool IsInstallable => Variants.Count > 0 && Level is CatalogLevel.Verified or CatalogLevel.Partial;

    /// <summary>The recommended variant, else the first; <see langword="null"/> without variants.</summary>
    public CatalogVariant? DefaultVariant => Variants.FirstOrDefault(variant => variant.Recommended) ?? Variants.FirstOrDefault();

    /// <summary>The companions <paramref name="variant"/> installs, in the variant's order.</summary>
    /// <param name="variant">A variant of this entry.</param>
    /// <exception cref="InvalidOperationException">The variant names a companion the entry does not have.</exception>
    public IReadOnlyList<CatalogCompanion> CompanionsOf(CatalogVariant variant)
    {
        ArgumentNullException.ThrowIfNull(variant);
        return [.. variant.Companions.Select(id => Companions.FirstOrDefault(companion => companion.Id == id)
            ?? throw new InvalidOperationException($"{Id}: variant {variant.Id} names the unknown companion '{id}'."))];
    }
}
