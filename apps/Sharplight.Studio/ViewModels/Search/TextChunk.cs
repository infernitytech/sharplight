namespace Sharplight.Studio.ViewModels.Search;

/// <summary>A passage of a document.</summary>
/// <param name="Start">Where it starts in the document's text.</param>
/// <param name="Text">The passage.</param>
internal sealed record TextChunk(int Start, string Text);
