using Sharplight.ModelFeed.Signing;

namespace Sharplight.ModelFeed.Client;

/// <summary>Trust and limits of a <see cref="FeedClient"/>.</summary>
public sealed class FeedClientOptions
{
    /// <summary>Trusted root keys; the feed's key set must be signed by one of them.</summary>
    public IList<FeedPublicKey> TrustedRoots { get; set; } = [];

    /// <summary>
    /// Whether signatures are required (default). When <see langword="false"/>, signatures are not checked and the snapshot is
    /// marked unverified (development and customer feeds accepted by policy); digests and hashes are always checked.
    /// </summary>
    public bool RequireSignature { get; set; } = true;

    /// <summary>Whether an expired index is accepted with a warning (offline sites with an imported mirror) instead of rejected.</summary>
    public bool AllowExpiredIndex { get; set; }

    /// <summary>Largest accepted feed document (index, page, manifest) in bytes.</summary>
    public long MaxDocumentBytes { get; set; } = 16L << 20;

    /// <summary>Allowed clock difference for times in the future (index update time).</summary>
    public TimeSpan ClockSkew { get; set; } = TimeSpan.FromMinutes(10);
}
