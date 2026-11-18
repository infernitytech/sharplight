using Sharplight.ModelFeed.Client;
using Sharplight.ModelFeed.Resolution;
using Sharplight.ModelFeed.Signing;

namespace Sharplight.Studio.Services.Feed;

/// <summary>Everything <see cref="FeedService"/> is configured with besides the user's settings.</summary>
internal sealed record FeedOptions
{
    /// <summary>The address of the official feed.</summary>
    public const string DefaultUrl = "https://download.sharplight.dev/models";

    /// <summary>The stable channel.</summary>
    public const string Stable = "stable";

    /// <summary>The unstable channel.</summary>
    public const string Unstable = "unstable";

    /// <summary>The name of the embedded resource that holds the official feed's root key.</summary>
    public const string RootKeyResource = "Sharplight.Studio.sharplight-feed.pub.pem";

    /// <summary>The root key the official feed is signed with (<c>tools/model-feed/keys/sharplight-feed.pub.pem</c>, embedded in the app).</summary>
    public static FeedPublicKey OfficialRootKey()
    {
        using var stream = typeof(FeedOptions).Assembly.GetManifestResourceStream(RootKeyResource)
            ?? throw new InvalidOperationException($"The embedded resource {RootKeyResource} is missing.");
        using var reader = new StreamReader(stream);
        return FeedPublicKey.Parse(reader.ReadToEnd());
    }

    /// <summary>The root keys that sign the feed's key set.</summary>
    public required IReadOnlyList<FeedPublicKey> TrustedRoots { get; init; }

    /// <summary>The file that remembers the accepted index sequences (rollback protection).</summary>
    public required string StateFile { get; init; }

    /// <summary>The folder that keeps the last verified documents of every feed and channel, for offline use.</summary>
    public required string CacheDirectory { get; init; }

    /// <summary>What the packages' requirements are checked against.</summary>
    public CompatibilityEnvironment Environment { get; init; } = new();

    /// <summary>Creates the network or folder transport of a feed address; null uses HTTP(S) and folders.</summary>
    public Func<string, IFeedTransport>? TransportFactory { get; init; }

    /// <summary>The clock, or null for the system clock.</summary>
    public TimeProvider? Time { get; init; }
}
