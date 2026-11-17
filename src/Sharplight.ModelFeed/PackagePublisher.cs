namespace Sharplight.ModelFeed;

/// <summary>Who made the model (not who packaged it).</summary>
public sealed record PackagePublisher
{
    /// <summary>Organization or person.</summary>
    public string? Name { get; init; }

    /// <summary>ISO 3166-1 alpha-2 country code of the publisher.</summary>
    public string? CountryOfOrigin { get; init; }

    /// <summary>Publisher web page.</summary>
    public string? Url { get; init; }
}
