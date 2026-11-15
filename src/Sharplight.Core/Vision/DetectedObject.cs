namespace Sharplight.Vision;

/// <summary>One object found by an <see cref="IObjectDetector"/>.</summary>
/// <param name="ClassIndex">The class index in the model's label list.</param>
/// <param name="Label">The class name.</param>
/// <param name="Score">The confidence in 0..1.</param>
/// <param name="Box">The box in pixels of the input image.</param>
public sealed record DetectedObject(int ClassIndex, string Label, float Score, BoundingBox Box);
