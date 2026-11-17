namespace Sharplight.Studio.Services.Updates;

/// <summary>Maps between the channel names of the feed (<c>linux-x64-stable</c>, <c>linux-x64-unstable</c>) and <see cref="UpdateChannelKind"/>.</summary>
internal static class UpdateChannels
{
    private const string StableSuffix = "-stable";
    private const string PreviewSuffix = "-unstable";

    /// <summary>The kind of an installed channel name; stable when the name is null or has no known suffix.</summary>
    public static UpdateChannelKind KindOf(string? channel) =>
        channel is not null && channel.EndsWith(PreviewSuffix, StringComparison.Ordinal) ? UpdateChannelKind.Preview : UpdateChannelKind.Stable;

    /// <summary>The channel name of <paramref name="kind"/> on the platform of <paramref name="installed"/>; null when nothing is installed.</summary>
    public static string? Name(string? installed, UpdateChannelKind kind)
    {
        if (installed is null)
        {
            return null;
        }

        var platform = installed;
        foreach (var suffix in new[] { StableSuffix, PreviewSuffix })
        {
            if (platform.EndsWith(suffix, StringComparison.Ordinal))
            {
                platform = platform[..^suffix.Length];
                break;
            }
        }

        return platform + (kind == UpdateChannelKind.Preview ? PreviewSuffix : StableSuffix);
    }

    /// <summary>The settings value of <paramref name="kind"/>.</summary>
    public static string Key(UpdateChannelKind kind) => kind == UpdateChannelKind.Preview ? "unstable" : "stable";

    /// <summary>The kind stored in the settings; null (follow the installed channel) for null or anything unknown.</summary>
    public static UpdateChannelKind? Parse(string? key) => key switch
    {
        "stable" => UpdateChannelKind.Stable,
        "unstable" => UpdateChannelKind.Preview,
        _ => null,
    };
}
