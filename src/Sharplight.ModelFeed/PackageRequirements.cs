namespace Sharplight.ModelFeed;

/// <summary>What a host needs to install and load a package.</summary>
public sealed record PackageRequirements
{
    /// <summary>Minimum host (server) version as a version range (<c>&gt;=1.0.0</c>), or <see langword="null"/>.</summary>
    public string? Server { get; init; }

    /// <summary>Minimum Sharplight engine version as a version range.</summary>
    public string? Engine { get; init; }

    /// <summary><c>.slm</c> container version.</summary>
    public int? PackageFormat { get; init; }

    /// <summary>Neutral engine feature names a device must have.</summary>
    public IReadOnlyList<string> DeviceFeatures { get; init; } = [];

    /// <summary>Memory needed per device kind (first-pass placement filter).</summary>
    public IReadOnlyList<MemoryRequirement> Memory { get; init; } = [];

    /// <summary>Disk bytes needed.</summary>
    public long? Disk { get; init; }
}
