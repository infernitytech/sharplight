namespace Sharplight.ModelFeed;

/// <summary>The engine and tool that built a package.</summary>
public sealed record ConverterInfo
{
    /// <summary>Engine version.</summary>
    public string? Engine { get; init; }

    /// <summary>Tool and version.</summary>
    public string? Tool { get; init; }

    /// <summary>Conversion options.</summary>
    public IReadOnlyDictionary<string, string> Options { get; init; } = new Dictionary<string, string>();
}
