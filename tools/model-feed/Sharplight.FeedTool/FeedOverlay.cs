namespace Sharplight.FeedTool;

/// <summary>
/// Feed-only data that is not part of the verified catalog (<c>tools/model-feed/feed-overlay.json</c>): dependencies between packages,
/// summaries, minimum server versions and listing state, by package id.
/// </summary>
internal sealed record FeedOverlay
{
    /// <summary>Schema version (1).</summary>
    public int SchemaVersion { get; init; } = 1;

    /// <summary>Overrides by package id.</summary>
    public IReadOnlyDictionary<string, FeedOverlayPackage> Packages { get; init; } = new Dictionary<string, FeedOverlayPackage>();
}
