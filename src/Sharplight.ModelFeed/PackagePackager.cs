namespace Sharplight.ModelFeed;

/// <summary>Who converted and signed the package.</summary>
public sealed record PackagePackager
{
    /// <summary>Packager name.</summary>
    public string? Name { get; init; }

    /// <summary>The feed the packager publishes to.</summary>
    public string? Feed { get; init; }
}
