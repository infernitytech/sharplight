namespace Sharplight.Studio.Models;

/// <summary>One timestamped segment of a stored transcript.</summary>
/// <param name="Start">Start, in seconds from the beginning of the audio.</param>
/// <param name="End">End, in seconds.</param>
/// <param name="Text">The spoken text.</param>
internal sealed record TranscriptSegmentRecord(double Start, double End, string Text);
