namespace Sharplight.ModelFeed;

/// <summary>A set of portable device code inside a package.</summary>
public sealed record PortableCodeSet
{
    /// <summary>Execution provider.</summary>
    public required string Provider { get; init; }

    /// <summary>Device profile.</summary>
    public string? Profile { get; init; }

    /// <summary>Code set id.</summary>
    public string? SetId { get; init; }
}
