namespace Sharplight.ModelFeed;

/// <summary>The channels of the official feed: new versions go to <see cref="Unstable"/>, tested ones are promoted to <see cref="Stable"/>.</summary>
public static class FeedChannels
{
    /// <summary>Promoted, tested versions (the default for hosts).</summary>
    public const string Stable = "stable";

    /// <summary>Every published version.</summary>
    public const string Unstable = "unstable";
}
