namespace Sharplight.ModelFeed.Index;

/// <summary>One version in a package index: its manifest digest, current state and what resolution needs.</summary>
public sealed record FeedPackageVersion
{
    /// <summary>Version.</summary>
    public required string Version { get; init; }

    /// <summary>Digest of <c>manifest.json</c>.</summary>
    public required string ManifestDigest { get; init; }

    /// <summary>Package file size.</summary>
    public long Size { get; init; }

    /// <summary>When the version was published.</summary>
    public DateTimeOffset Published { get; init; }

    /// <summary>Current state (authoritative; the manifest keeps the state at publication).</summary>
    public PackageState State { get; init; } = new();

    /// <summary>Requirements (copy from the manifest).</summary>
    public PackageRequirements? Requirements { get; init; }

    /// <summary>Dependencies (copy from the manifest).</summary>
    public IReadOnlyList<PackageDependency> Dependencies { get; init; } = [];
}
