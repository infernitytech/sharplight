namespace Sharplight.ModelFeed.Resolution;

/// <summary>One available version of a package as resolution sees it.</summary>
public sealed record PackageCandidate
{
    /// <summary>Package id.</summary>
    public required string Id { get; init; }

    /// <summary>Version.</summary>
    public required string Version { get; init; }

    /// <summary>Requirements, or <see langword="null"/>.</summary>
    public PackageRequirements? Requirements { get; init; }

    /// <summary>Dependencies.</summary>
    public IReadOnlyList<PackageDependency> Dependencies { get; init; } = [];

    /// <summary>Current state (yanked versions are never chosen; unlisted ones only when pinned exactly).</summary>
    public PackageState? State { get; init; }
}
