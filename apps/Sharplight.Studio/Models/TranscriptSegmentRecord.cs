namespace Sharplight.Studio.Models;

/// <summary>One timestamped segment of a stored transcript.</summary>
/// <param name="Start">Start, in seconds from the beginning of the audio.</param>
/// <param name="End">End, in seconds.</param>
/// <param name="Text">The spoken text.</param>
/// <param name="Speaker">The speaker label of a diarizing model (<c>0</c>, <c>Speaker 1</c>), or <see langword="null"/>.</param>
internal sealed record TranscriptSegmentRecord(double Start, double End, string Text, string? Speaker = null)
{
    /// <summary>The label shown and exported: a bare number becomes <c>Speaker N</c>; empty when the segment has no speaker.</summary>
    public string SpeakerLabel => string.IsNullOrWhiteSpace(Speaker) ? "" : int.TryParse(Speaker.Trim(), out int number) ? $"Speaker {number}" : Speaker.Trim();
}
