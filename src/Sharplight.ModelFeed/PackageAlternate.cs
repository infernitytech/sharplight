namespace Sharplight.ModelFeed;

/// <summary>A suggested replacement package.</summary>
public sealed record PackageAlternate
{
    /// <summary>Package id.</summary>
    public required string Id { get; init; }

    /// <summary>Version range.</summary>
    public string? Range { get; init; }
}
