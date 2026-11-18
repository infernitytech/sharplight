using Sharplight.Studio.Services.Feed;

namespace Sharplight.Studio.ViewModels.Settings;

/// <summary>One entry of the model feed channel selector.</summary>
/// <param name="Value">The channel name in the feed (<c>stable</c>, <c>unstable</c>).</param>
/// <param name="Title">The name shown to the user.</param>
internal sealed record FeedChannelChoice(string Value, string Title)
{
    /// <summary>The channels the user can pick.</summary>
    public static IReadOnlyList<FeedChannelChoice> All { get; } =
    [
        new(FeedOptions.Stable, "Stable"),
        new(FeedOptions.Unstable, "Unstable (new packages first)"),
    ];

    /// <summary>The choice for a channel setting (null and unknown names are stable).</summary>
    public static FeedChannelChoice For(string? value) => All.FirstOrDefault(choice => string.Equals(choice.Value, value, StringComparison.OrdinalIgnoreCase)) ?? All[0];
}
