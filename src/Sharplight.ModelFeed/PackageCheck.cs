namespace Sharplight.ModelFeed;

/// <summary>Check of the built package file.</summary>
public sealed record PackageCheck
{
    /// <summary>Whether the package opened.</summary>
    public bool Loaded { get; init; }

    /// <summary>Hash of a smoke run's output, when one ran.</summary>
    public string? SmokeOutputSha256 { get; init; }
}
