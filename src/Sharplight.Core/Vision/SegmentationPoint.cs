namespace Sharplight.Vision;

/// <summary>A point prompt of <see cref="SegmentationPrompt"/> in pixels of the image (origin top left).</summary>
/// <param name="X">Horizontal position.</param>
/// <param name="Y">Vertical position.</param>
/// <param name="IsPositive"><see langword="true"/> when the point lies on the object, <see langword="false"/> when it lies on the background.</param>
public readonly record struct SegmentationPoint(float X, float Y, bool IsPositive = true);
