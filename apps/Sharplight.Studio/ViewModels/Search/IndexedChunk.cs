namespace Sharplight.Studio.ViewModels.Search;

/// <summary>A passage of an indexed document with its embedding (normalized to length 1, so a dot product is the cosine similarity).</summary>
/// <param name="File">The document's path.</param>
/// <param name="Start">Where the passage starts in the section's text.</param>
/// <param name="Text">The passage.</param>
/// <param name="Vector">Its embedding.</param>
/// <param name="Page">The page, slide or sheet number (1-based), or 0.</param>
/// <param name="Line">The line of the file the passage starts at (1-based), or 0.</param>
/// <param name="Heading">The heading it stands under, or empty.</param>
internal sealed record IndexedChunk(string File, int Start, string Text, float[] Vector, int Page = 0, int Line = 0, string Heading = "");
