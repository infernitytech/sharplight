namespace Sharplight.Studio.Models;

/// <summary>A point of a segmentation prompt as the vision page stores it: pixels of the input image.</summary>
/// <param name="X">Horizontal position.</param>
/// <param name="Y">Vertical position.</param>
/// <param name="Positive">Whether the point lies on the object.</param>
internal sealed record VisionPointRecord(float X, float Y, bool Positive);
