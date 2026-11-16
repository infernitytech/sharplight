namespace Sharplight.Vision;

/// <summary>Per-request settings of <see cref="ISemanticSegmenter"/>.</summary>
public sealed class SemanticSegmentationOptions
{
    /// <summary>Whether the label map has the size of the input image (class scores resampled bilinearly, as transformers' post-processing does); otherwise the model's output resolution.</summary>
    public bool ResizeToInput { get; set; } = true;

    /// <summary>Whether the result carries the raw class scores at the model's output resolution (<see cref="SemanticSegmentation.Logits"/>).</summary>
    public bool IncludeLogits { get; set; }
}
