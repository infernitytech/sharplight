namespace Sharplight.DesignTool;

/// <summary>A set of icons drawn on one grid.</summary>
/// <param name="Name">The set's name (<c>icon</c>, <c>caption</c>).</param>
/// <param name="Size">Grid size in px (the view box).</param>
/// <param name="Comment">The description written above the icons in Studio's resource dictionary.</param>
internal sealed record IconSetDefinition(string Name, int Size, string Comment);
