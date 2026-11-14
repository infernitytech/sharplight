namespace Sharplight.Studio.Models;

/// <summary>An accent colour scheme: the base colour and its lighter and darker steps (hex <c>#RRGGBB</c>).</summary>
/// <param name="Name">Key stored in the settings.</param>
/// <param name="Accent">The base accent.</param>
/// <param name="Light1">One step lighter.</param>
/// <param name="Light2">Two steps lighter.</param>
/// <param name="Light3">Three steps lighter.</param>
/// <param name="Dark1">One step darker.</param>
/// <param name="Dark2">Two steps darker.</param>
/// <param name="Dark3">Three steps darker.</param>
/// <param name="Secondary">Second colour of gradients and the second chart series.</param>
internal sealed record AccentPalette(string Name, string Accent, string Light1, string Light2, string Light3, string Dark1, string Dark2, string Dark3, string Secondary)
{
    /// <summary>The palettes offered in the settings.</summary>
    public static IReadOnlyList<AccentPalette> All { get; } =
    [
        new("Violet", "#7C5CFF", "#9479FF", "#AD97FF", "#C7B8FF", "#6646E6", "#5133C2", "#3D2499", "#22C1EE"),
        new("Ocean", "#2F80ED", "#4F95F0", "#76AEF4", "#A2C8F8", "#1F6AD0", "#1754AA", "#103F82", "#2DD4BF"),
        new("Emerald", "#10B981", "#34C796", "#5ED5AC", "#8FE3C6", "#0C9A6B", "#097A55", "#065C40", "#3B82F6"),
        new("Amber", "#F59E0B", "#F7B036", "#F9C465", "#FBD898", "#D18406", "#A66904", "#7C4E03", "#EF4444"),
        new("Rose", "#F43F5E", "#F6627C", "#F8879B", "#FBB0BE", "#D72A49", "#AE1F3A", "#83162B", "#A855F7"),
    ];

    /// <summary>The palette called <paramref name="name"/>, or the first.</summary>
    public static AccentPalette Get(string? name) => All.FirstOrDefault(palette => palette.Name == name) ?? All[0];
}
