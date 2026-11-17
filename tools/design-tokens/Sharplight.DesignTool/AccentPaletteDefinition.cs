namespace Sharplight.DesignTool;

/// <summary>One accent palette of the tokens.</summary>
/// <param name="Key">Kebab-case key (<c>violet</c>), used as <c>data-accent</c> in CSS.</param>
/// <param name="Name">Display name and settings key (<c>Violet</c>).</param>
/// <param name="Accent">The base colour.</param>
/// <param name="Light">Three lighter steps.</param>
/// <param name="Dark">Three darker steps.</param>
/// <param name="Secondary">The secondary colour.</param>
internal sealed record AccentPaletteDefinition(string Key, string Name, string Accent, IReadOnlyList<string> Light, IReadOnlyList<string> Dark, string Secondary);
