namespace Sharplight.ModelFeed;

/// <summary>The manifest of one package version (<c>manifest.json</c>, docs/model-feed.md): what the model is, who made and packaged it, its license, requirements, dependencies, provenance and files. Readers ignore unknown fields.</summary>
public sealed record PackageManifest
{
    /// <summary>Manifest schema version (1); changes only for breaking changes.</summary>
    public int SchemaVersion { get; init; } = 1;

    /// <summary>Package id: lower case <c>[a-z0-9.-]</c>, <c>&lt;catalog entry&gt;.&lt;variant&gt;</c>; never reused.</summary>
    public required string Id { get; init; }

    /// <summary>Semantic version (SemVer 2); a published version is immutable.</summary>
    public required string Version { get; init; }

    /// <summary>Names and summary for listings.</summary>
    public PackageDisplay? Display { get; init; }

    /// <summary>The model this package is a variant of, or <see langword="null"/> for a package outside any group (docs/model-feed.md "Model groups").</summary>
    public PackageGroup? Group { get; init; }

    /// <summary>Who made the model.</summary>
    public PackagePublisher? Publisher { get; init; }

    /// <summary>Who converted and signed the package.</summary>
    public PackagePackager? Packager { get; init; }

    /// <summary>License of the model.</summary>
    public PackageLicense? License { get; init; }

    /// <summary>What the model is and does.</summary>
    public PackageModel? Model { get; init; }

    /// <summary>Minimum versions, memory and disk.</summary>
    public PackageRequirements? Requirements { get; init; }

    /// <summary>Other packages (version ranges); companions merged into the package are not dependencies.</summary>
    public IReadOnlyList<PackageDependency> Dependencies { get; init; } = [];

    /// <summary>Where the package was built from.</summary>
    public PackageProvenance? Provenance { get; init; }

    /// <summary>How the model was verified.</summary>
    public PackageVerification? Verification { get; init; }

    /// <summary>The package's files with sizes and hashes (one with role <c>package</c>: the <c>.slm</c>).</summary>
    public IReadOnlyList<PackageFile> Files { get; init; } = [];

    /// <summary>Portable device code sets inside the package.</summary>
    public IReadOnlyList<PortableCodeSet> PortableCode { get; init; } = [];

    /// <summary>Listing state at publication (the feed's package index carries the current state).</summary>
    public PackageState? State { get; init; }

    /// <summary>Free text notes.</summary>
    public string? Notes { get; init; }

    /// <summary>
    /// The group of the package: <see cref="Group"/>, else one derived from the provenance (catalog entry and variant) of manifests
    /// written before groups existed, else <see langword="null"/>.
    /// </summary>
    public PackageGroup? EffectiveGroup => Group ?? (Provenance is { CatalogEntry: { Length: > 0 } entry, CatalogVariant: { Length: > 0 } variant } && Id == $"{entry}.{variant}"
        ? new PackageGroup { Id = entry, Name = Display?.Name, Variant = variant, Label = Display?.Variant, Quantization = Model?.Quantization, Size = PrimaryFile?.Size ?? 0 }
        : null);

    /// <summary>The <c>.slm</c> file (role <see cref="PackageFileRoles.Package"/>), or <see langword="null"/>.</summary>
    public PackageFile? PrimaryFile => Files.FirstOrDefault(file => file.Role is null or PackageFileRoles.Package);
}
