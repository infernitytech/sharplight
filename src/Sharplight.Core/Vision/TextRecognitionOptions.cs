namespace Sharplight.Vision;

/// <summary>Per-request settings of <see cref="ITextRecognizer"/>.</summary>
public sealed class TextRecognitionOptions
{
    /// <summary>The most tokens the decoder generates (the text of one image).</summary>
    public int MaxTokens { get; set; } = 128;
}
