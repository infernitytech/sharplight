namespace Sharplight.Studio.Services.Platform;

/// <summary>A file type filter of a file picker.</summary>
/// <param name="Name">Shown in the picker.</param>
/// <param name="Patterns">Glob patterns such as <c>*.pdf</c>.</param>
internal sealed record FileFilter(string Name, IReadOnlyList<string> Patterns);
