namespace Sharplight.ModelFeed.Index;

/// <summary>What a listing needs about one package without further requests (fields of its latest listed version).</summary>
public sealed record FeedPackageSummary
{
    /// <summary>Package id.</summary>
    public required string Id { get; init; }

    /// <summary>Latest listed, not yanked version (prereleases only when no release exists).</summary>
    public required string LatestVersion { get; init; }

    /// <summary>All versions, ascending.</summary>
    public IReadOnlyList<string> Versions { get; init; } = [];

    /// <summary>Display data.</summary>
    public PackageDisplay? Display { get; init; }

    /// <summary>The model this package is a variant of (the manifest's group, or one derived from its provenance), or <see langword="null"/>.</summary>
    public PackageGroup? Group { get; init; }

    /// <summary>Publisher and country of origin.</summary>
    public PackagePublisher? Publisher { get; init; }

    /// <summary>License.</summary>
    public PackageLicense? License { get; init; }

    /// <summary>Kind, tasks, capabilities, size data.</summary>
    public PackageModel? Model { get; init; }

    /// <summary>Requirements.</summary>
    public PackageRequirements? Requirements { get; init; }

    /// <summary>Verification level, reference, providers and date of the latest version.</summary>
    public PackageVerification? Verification { get; init; }

    /// <summary>Dependencies.</summary>
    public IReadOnlyList<PackageDependency> Dependencies { get; init; } = [];

    /// <summary>Package file size in bytes.</summary>
    public long Size { get; init; }

    /// <summary>Current state of the latest version.</summary>
    public PackageState? State { get; init; }

    /// <summary>Digest of the latest version's manifest.</summary>
    public required string ManifestDigest { get; init; }

    /// <summary>Digest of the package index (<c>packages/&lt;id&gt;/index.json</c>).</summary>
    public required string IndexDigest { get; init; }
}
