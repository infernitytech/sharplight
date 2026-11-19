namespace Sharplight.Vision;

/// <summary>One region of a <see cref="DocumentPage"/>: a piece of layout and the text read from it.</summary>
/// <param name="Order">The position in reading order (0 is first).</param>
/// <param name="Label">The layout class of the region (the layout model's label, e.g. <c>text</c>, <c>table</c>, <c>doc_title</c>).</param>
/// <param name="Box">Where the region is, in pixels of the page image.</param>
/// <param name="Score">The layout model's confidence in 0..1.</param>
/// <param name="Text">The recognized text, or <see langword="null"/> for regions that are not read (figures).</param>
/// <param name="Task">The recognition task the region was read with (<see cref="TextRecognitionOptions.Task"/>), or <see langword="null"/> when it was not read or the recognizer has no tasks.</param>
public sealed record DocumentRegion(int Order, string Label, BoundingBox Box, float Score, string? Text, string? Task);
