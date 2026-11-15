namespace Sharplight.Vision;

/// <summary>The result of <see cref="IObjectDetector.DetectAsync"/>.</summary>
/// <param name="Objects">The detected objects, best score first.</param>
public sealed record ObjectDetection(IReadOnlyList<DetectedObject> Objects);
