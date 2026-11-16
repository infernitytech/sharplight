namespace Sharplight.Vision;

/// <summary>One piece of recognized text of a <see cref="TextRecognition"/>.</summary>
/// <param name="Text">The text.</param>
/// <param name="Box">Where the text is in pixels of the image, or <see langword="null"/> when the model does not locate it (line recognizers read the whole image).</param>
public sealed record TextRegion(string Text, BoundingBox? Box);
