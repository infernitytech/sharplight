namespace Sharplight.Vision;

/// <summary>Per-request settings of <see cref="IImageClassifier"/> and zero-shot classification.</summary>
public sealed class ImageClassificationOptions
{
    /// <summary>How many of the best classes to return; 0 returns every class.</summary>
    public int TopK { get; set; } = 5;
}
