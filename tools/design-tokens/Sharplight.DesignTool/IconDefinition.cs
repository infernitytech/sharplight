namespace Sharplight.DesignTool;

/// <summary>One icon of design/icons.json.</summary>
/// <param name="Name">Kebab-case name (<c>chevron-down</c>).</param>
/// <param name="Set">The set: <c>icon</c> (24 x 24) or <c>caption</c> (10 x 10).</param>
/// <param name="Path">SVG path data.</param>
internal sealed record IconDefinition(string Name, string Set, string Path);
