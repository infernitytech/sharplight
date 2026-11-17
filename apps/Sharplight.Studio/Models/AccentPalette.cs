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
    /// <summary>The palettes offered in the settings (from the design tokens, design/tokens.json).</summary>
    public static IReadOnlyList<AccentPalette> All => GeneratedAccentPalettes.All;

    /// <summary>The palette called <paramref name="name"/>, or the first.</summary>
    public static AccentPalette Get(string? name) => All.FirstOrDefault(palette => palette.Name == name) ?? All[0];
}
