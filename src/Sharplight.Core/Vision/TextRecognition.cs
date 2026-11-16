namespace Sharplight.Vision;

/// <summary>The result of <see cref="ITextRecognizer.RecognizeAsync"/>.</summary>
/// <param name="Text">The recognized text of the whole image (the regions' texts joined by line breaks).</param>
/// <param name="Regions">The pieces of text in reading order; with their boxes when the model locates text.</param>
public sealed record TextRecognition(string Text, IReadOnlyList<TextRegion> Regions);
