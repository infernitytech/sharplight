namespace Sharplight.Studio.Models;

/// <summary>One candidate mask of a segmentation as the vision page stores it.</summary>
/// <param name="Index">The candidate's number, from 1.</param>
/// <param name="Score">The model's predicted IoU of the mask, 0 to 1.</param>
/// <param name="Area">The number of pixels inside the mask.</param>
internal sealed record VisionMaskRecord(int Index, float Score, int Area);
