namespace Sharplight.ModelFeed;

/// <summary>Listing state of a package version.</summary>
public sealed record PackageState
{
    /// <summary>Whether listings show the version.</summary>
    public bool Listed { get; init; } = true;

    /// <summary>Deprecation (still installable, flagged).</summary>
    public PackageDeprecation? Deprecated { get; init; }

    /// <summary>Yank (not installable; installed copies are flagged).</summary>
    public PackageYank? Yanked { get; init; }
}
