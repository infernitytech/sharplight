namespace Sharplight.Vision;

/// <summary>The result of <see cref="IObjectDetector.DetectAsync"/>.</summary>
/// <param name="Objects">The detected objects, best score first; in reading order (<see cref="DetectedObject.ReadingOrder"/>) for models that predict one.</param>
public sealed record ObjectDetection(IReadOnlyList<DetectedObject> Objects);
