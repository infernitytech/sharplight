namespace Sharplight.Vision;

/// <summary>Per-request settings of <see cref="IImageSegmenter"/>.</summary>
public sealed class SegmentationOptions
{
    /// <summary>
    /// Whether every candidate mask of the model is returned (an ambiguous prompt such as one point has several valid answers: part, object, group);
    /// otherwise only the best one (the single-mask candidate when it is stable, else the candidate with the highest predicted IoU).
    /// </summary>
    public bool MultipleMasks { get; set; } = true;
}
