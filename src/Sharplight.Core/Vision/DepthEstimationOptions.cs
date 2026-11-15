namespace Sharplight.Vision;

/// <summary>Per-request settings of <see cref="IDepthEstimator"/>.</summary>
public sealed class DepthEstimationOptions
{
    /// <summary>Whether the map is resampled (bilinear) to the size of the input image; otherwise it has the model's output resolution.</summary>
    public bool ResizeToInput { get; set; } = true;
}
