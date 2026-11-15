namespace Sharplight.Speech;

/// <summary>How fine the timestamps of a transcription are.</summary>
public enum TimestampGranularity
{
    /// <summary>Segment start and end times only.</summary>
    Segment,

    /// <summary>Segment times plus per-word times (<see cref="TranscriptionSegment.Words"/>), where the model supports it.</summary>
    Word,
}
