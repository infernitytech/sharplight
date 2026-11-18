namespace Sharplight.Studio.ViewModels.Search;

/// <summary>A passage of an indexed document with its embedding (normalized to length 1, so a dot product is the cosine similarity).</summary>
/// <param name="File">The document's path.</param>
/// <param name="Start">Where the passage starts in the document's text.</param>
/// <param name="Text">The passage.</param>
/// <param name="Vector">Its embedding.</param>
internal sealed record IndexedChunk(string File, int Start, string Text, float[] Vector);
