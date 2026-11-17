namespace Sharplight.DesignTool;

/// <summary>A token: its kebab-case name within its group and its value as written in CSS (strings, numbers, joined font stacks).</summary>
/// <param name="Name">The name (<c>surface-alt</c>).</param>
/// <param name="Value">The value (<c>#1E222B</c>, <c>8px</c>, <c>0.16</c>).</param>
internal sealed record TokenValue(string Name, string Value);
