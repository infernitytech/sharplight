namespace Sharplight.Vision;

/// <summary>Per-request settings of <see cref="IObjectDetector"/>.</summary>
public sealed class ObjectDetectionOptions
{
    /// <summary>Objects scoring below this are dropped.</summary>
    public float ScoreThreshold { get; set; } = 0.5f;

    /// <summary>At most this many objects are returned (best first); 0 for no limit.</summary>
    public int MaxDetections { get; set; } = 100;

    /// <summary>
    /// Non-maximum suppression threshold (intersection over union), or <see langword="null"/> for none. Set-based detectors (DETR, RT-DETR) need none;
    /// the setting serves models that emit overlapping boxes.
    /// </summary>
    public float? NmsIouThreshold { get; set; }

    /// <summary>Whether suppression also removes overlapping boxes of other classes.</summary>
    public bool ClassAgnosticNms { get; set; }

    /// <summary>
    /// Whether the objects of models that predict masks carry them (<see cref="DetectedObject.Mask"/>, at the size of the image: one bool per pixel). Turn it off when only
    /// the boxes are needed (page-sized masks of many objects are large).
    /// </summary>
    public bool IncludeMasks { get; set; } = true;
}
