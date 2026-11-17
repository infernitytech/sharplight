namespace Sharplight.DesignTool;

/// <summary>One output of the generator.</summary>
/// <param name="RelativePath">Path below the repository root, with forward slashes.</param>
/// <param name="Content">The file's text.</param>
internal sealed record GeneratedFile(string RelativePath, string Content);
