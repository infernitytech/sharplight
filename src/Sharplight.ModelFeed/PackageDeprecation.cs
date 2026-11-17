namespace Sharplight.ModelFeed;

/// <summary>Why a package version is deprecated.</summary>
public sealed record PackageDeprecation
{
    /// <summary>When.</summary>
    public DateTimeOffset Date { get; init; }

    /// <summary>Why.</summary>
    public string? Reason { get; init; }

    /// <summary>Suggested replacement.</summary>
    public PackageAlternate? Alternate { get; init; }
}
