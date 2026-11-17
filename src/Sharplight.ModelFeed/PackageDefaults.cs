namespace Sharplight.ModelFeed;

/// <summary>Recommended generation defaults.</summary>
public sealed record PackageDefaults
{
    /// <summary>Sampling temperature.</summary>
    public double? Temperature { get; init; }

    /// <summary>Nucleus sampling probability.</summary>
    public double? TopP { get; init; }

    /// <summary><c>embedded</c> when the package carries its chat template.</summary>
    public string? ChatTemplate { get; init; }
}
