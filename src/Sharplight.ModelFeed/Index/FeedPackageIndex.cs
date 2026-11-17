namespace Sharplight.ModelFeed.Index;

/// <summary>All versions of one package (<c>packages/&lt;id&gt;/index.json</c>).</summary>
public sealed record FeedPackageIndex
{
    /// <summary>Schema version (1).</summary>
    public int SchemaVersion { get; init; } = 1;

    /// <summary>Package id.</summary>
    public required string Id { get; init; }

    /// <summary>Versions, ascending.</summary>
    public IReadOnlyList<FeedPackageVersion> Versions { get; init; } = [];
}
