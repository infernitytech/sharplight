namespace Sharplight.Studio.Services.Search;

/// <summary>A piece of a passage, shown plain or highlighted as a match of the question.</summary>
/// <param name="Text">The characters.</param>
/// <param name="IsMatch">The piece is a word of the question.</param>
internal sealed record TextSegment(string Text, bool IsMatch);
