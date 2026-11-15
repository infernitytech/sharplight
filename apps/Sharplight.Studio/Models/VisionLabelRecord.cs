namespace Sharplight.Studio.Models;

/// <summary>One class of a classification as the vision page stores it.</summary>
/// <param name="Index">The class index.</param>
/// <param name="Label">The class name.</param>
/// <param name="Score">The probability or confidence, 0 to 1.</param>
internal sealed record VisionLabelRecord(int Index, string Label, float Score);
