using Sharplight.ModelFeed;

namespace Sharplight.FeedTool;

/// <summary>Feed-only data of one package.</summary>
internal sealed record FeedOverlayPackage
{
    /// <summary>One-paragraph summary for listings.</summary>
    public string? Summary { get; init; }

    /// <summary>Dependencies on other packages.</summary>
    public IReadOnlyList<PackageDependency> Dependencies { get; init; } = [];

    /// <summary>Minimum host version range, or <see langword="null"/>.</summary>
    public string? Server { get; init; }

    /// <summary>Languages (ISO 639-1).</summary>
    public IReadOnlyList<string> Languages { get; init; } = [];

    /// <summary>Whether listings show the package.</summary>
    public bool Listed { get; init; } = true;
}
