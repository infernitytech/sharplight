using Sharplight.ModelFeed.Versioning;

namespace Sharplight.ModelFeed.Resolution;

/// <summary>What the installing host offers, for <see cref="PackageCompatibility"/>.</summary>
public sealed record CompatibilityEnvironment
{
    /// <summary>Host (server) version, or <see langword="null"/> to skip the host check (Studio, tools).</summary>
    public SemanticVersion? HostVersion { get; init; }

    /// <summary>Engine version, or <see langword="null"/> to skip the engine check.</summary>
    public SemanticVersion? EngineVersion { get; init; }

    /// <summary>Highest <c>.slm</c> container version the engine reads, or <see langword="null"/> to skip.</summary>
    public int? MaxPackageFormat { get; init; }
}
