namespace Sharplight.ModelFeed;

/// <summary>Why a package version was withdrawn.</summary>
public sealed record PackageYank
{
    /// <summary>When.</summary>
    public DateTimeOffset Date { get; init; }

    /// <summary>Why.</summary>
    public string? Reason { get; init; }

    /// <summary><see cref="YankSeverities"/>.</summary>
    public string Severity { get; init; } = YankSeverities.Quality;
}
