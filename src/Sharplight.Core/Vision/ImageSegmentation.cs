namespace Sharplight.Vision;

/// <summary>The result of <see cref="IImageSegmenter.SegmentAsync(ImageInput, SegmentationPrompt, SegmentationOptions?, CancellationToken)"/>.</summary>
/// <param name="Masks">The candidate masks in the model's order.</param>
/// <param name="ObjectScore">The model's logit that the prompt hits an object (above 0: likely).</param>
public sealed record ImageSegmentation(IReadOnlyList<SegmentMask> Masks, float ObjectScore)
{
    /// <summary>The candidate with the highest predicted IoU.</summary>
    public SegmentMask Best => Masks.MaxBy(m => m.Score) ?? throw new InvalidOperationException("The segmentation has no masks.");
}
