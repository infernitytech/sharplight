using Sharplight.Studio.Services.Updates;

namespace Sharplight.Studio.ViewModels.Updates;

/// <summary>One entry of the channel selector.</summary>
/// <param name="Kind">The release stream.</param>
/// <param name="Title">The name shown to the user.</param>
internal sealed record UpdateChannelChoice(UpdateChannelKind Kind, string Title)
{
    /// <summary>The release stream the user can pick.</summary>
    public static IReadOnlyList<UpdateChannelChoice> All { get; } =
    [
        new(UpdateChannelKind.Stable, "Stable"),
        new(UpdateChannelKind.Preview, "Preview (unstable)"),
    ];

    /// <summary>The choice for <paramref name="kind"/>.</summary>
    public static UpdateChannelChoice For(UpdateChannelKind kind) => All.Single(choice => choice.Kind == kind);
}
