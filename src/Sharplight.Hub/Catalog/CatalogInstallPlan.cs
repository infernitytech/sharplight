namespace Sharplight.Hub.Catalog;

/// <summary>
/// What installing a catalog variant does: the variant's files and its companions' files (each pinned to its commit), stored under
/// <see cref="Reference"/> in the model store and converted to one package with the companions in their roles.
/// </summary>
public sealed record CatalogInstallPlan
{
    private CatalogInstallPlan(CatalogEntry entry, CatalogVariant variant, IReadOnlyList<CatalogCompanion> companions, string storeTag)
    {
        Entry = entry;
        Variant = variant;
        Companions = companions;
        StoreTag = storeTag;
    }

    /// <summary>The entry.</summary>
    public CatalogEntry Entry { get; }

    /// <summary>The variant.</summary>
    public CatalogVariant Variant { get; }

    /// <summary>The companions installed with the variant.</summary>
    public IReadOnlyList<CatalogCompanion> Companions { get; }

    /// <summary>
    /// The tag the model is stored under: the quantization for GGUF variants (as <c>sharplight pull repo:Q8_0</c> stores it), the variant id
    /// when another variant of the entry shares the repository, else empty.
    /// </summary>
    public string StoreTag { get; }

    /// <summary>The model store name, <c>repository[:tag]</c>.</summary>
    public ModelReference Reference => new(Variant.Repository, StoreTag.Length == 0 ? null : StoreTag);

    /// <summary><c>entry/variant</c>, recorded in the manifest (<see cref="ModelManifest.Catalog"/>).</summary>
    public string Key => $"{Entry.Id}/{Variant.Id}";

    /// <summary>Bytes to download (variant and companions).</summary>
    public long TotalBytes => Variant.TotalBytes + Companions.Sum(companion => companion.TotalBytes);

    /// <summary>Number of files to download.</summary>
    public int FileCount => Variant.Files.Count + Companions.Sum(companion => companion.Files.Count);

    /// <summary>The plan for <paramref name="variant"/> of <paramref name="entry"/>.</summary>
    /// <param name="entry">The entry.</param>
    /// <param name="variant">One of its variants.</param>
    /// <exception cref="ArgumentException">The variant is not the entry's, or names an unknown companion.</exception>
    public static CatalogInstallPlan For(CatalogEntry entry, CatalogVariant variant)
    {
        ArgumentNullException.ThrowIfNull(entry);
        ArgumentNullException.ThrowIfNull(variant);
        if (!entry.Variants.Contains(variant))
        {
            throw new ArgumentException($"Variant {variant.Id} is not a variant of {entry.Id}.", nameof(variant));
        }

        IReadOnlyList<CatalogCompanion> companions;
        try
        {
            companions = entry.CompanionsOf(variant);
        }
        catch (InvalidOperationException error)
        {
            throw new ArgumentException(error.Message, nameof(variant), error);
        }

        bool shared = entry.Variants.Any(other => !ReferenceEquals(other, variant) && string.Equals(other.Repository, variant.Repository, StringComparison.OrdinalIgnoreCase));
        string tag = variant.Format == CatalogFormats.Gguf && variant.Quantization is { Length: > 0 } quantization ? quantization
            : shared ? variant.Id
            : "";
        return new CatalogInstallPlan(entry, variant, companions, tag);
    }

    /// <summary>Whether <paramref name="manifest"/> is this plan installed: from the same catalog variant, or the same files at the same commit without companions.</summary>
    /// <param name="manifest">A stored model.</param>
    public bool IsInstalledAs(ModelManifest manifest)
    {
        ArgumentNullException.ThrowIfNull(manifest);
        if (manifest.Catalog is { } key)
        {
            return key == Key && manifest.Commit == Variant.Revision;
        }

        return Companions.Count == 0 && manifest.Commit == Variant.Revision
            && string.Equals(manifest.Repository, Variant.Repository, StringComparison.OrdinalIgnoreCase)
            && string.Equals(manifest.Tag, StoreTag, StringComparison.OrdinalIgnoreCase);
    }
}
