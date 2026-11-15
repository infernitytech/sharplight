namespace Sharplight.Studio.Models;

/// <summary>One detected object as the vision page stores it: the box in pixels of the input image.</summary>
/// <param name="ClassIndex">The class index.</param>
/// <param name="Label">The class name.</param>
/// <param name="Score">The confidence, 0 to 1.</param>
/// <param name="X1">Left edge.</param>
/// <param name="Y1">Top edge.</param>
/// <param name="X2">Right edge.</param>
/// <param name="Y2">Bottom edge.</param>
internal sealed record VisionBoxRecord(int ClassIndex, string Label, float Score, float X1, float Y1, float X2, float Y2);
