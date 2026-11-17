namespace Sharplight.ModelFeed;

/// <summary>Where a package was built from.</summary>
public sealed record PackageProvenance
{
    /// <summary>Verified-catalog entry id.</summary>
    public string? CatalogEntry { get; init; }

    /// <summary>Verified-catalog variant id.</summary>
    public string? CatalogVariant { get; init; }

    /// <summary>Upstream repositories at pinned revisions.</summary>
    public IReadOnlyList<ProvenanceSource> Sources { get; init; } = [];

    /// <summary>Engine and tool that converted the sources.</summary>
    public ConverterInfo? Converter { get; init; }

    /// <summary>When the package was built.</summary>
    public DateTimeOffset? Built { get; init; }
}
