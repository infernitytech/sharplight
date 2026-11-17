namespace Sharplight.ModelFeed;

/// <summary>A dependency on another package.</summary>
public sealed record PackageDependency
{
    /// <summary>Package id.</summary>
    public required string Id { get; init; }

    /// <summary>Allowed versions (<see cref="Versioning.VersionRange"/>); <see langword="null"/> allows all.</summary>
    public string? Range { get; init; }

    /// <summary><see cref="DependencyKinds.Required"/> (installed together) or <see cref="DependencyKinds.Optional"/> (offered).</summary>
    public string Kind { get; init; } = DependencyKinds.Required;

    /// <summary>Auxiliary role (<c>draft</c>, <c>voice-activity</c>, <c>lora</c>, ...) or <c>pairing</c>.</summary>
    public string? Role { get; init; }

    /// <summary>Whether the dependency must be installed together with the package.</summary>
    public bool IsRequired => string.Equals(Kind, DependencyKinds.Required, StringComparison.Ordinal);
}
