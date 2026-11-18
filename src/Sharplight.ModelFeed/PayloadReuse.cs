namespace Sharplight.ModelFeed;

/// <summary>
/// Staging-folder note (<see cref="FeedLayout.PayloadReuseFileName"/>) of a manifest-only version: its package file is the earlier
/// version's file (same SHA-256), stored once in the feed and copied under the new version's folder instead of being built and
/// uploaded again.
/// </summary>
public sealed record PayloadReuse
{
    /// <summary>The earlier version that holds the package file.</summary>
    public required string Version { get; init; }

    /// <summary>The package file name in the earlier version's folder.</summary>
    public required string File { get; init; }
}
