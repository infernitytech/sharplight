namespace Sharplight.Speech;

/// <summary>How fine the timestamps of a transcription are.</summary>
public enum TimestampGranularity
{
    /// <summary>
    /// No timestamps from the model: each window of audio becomes one segment (the default decoding of Hugging Face transformers; slightly
    /// faster). Segment times are the window bounds.
    /// </summary>
    None = -1,

    /// <summary>Segment start and end times only.</summary>
    Segment,

    /// <summary>Segment times plus per-word times (<see cref="TranscriptionSegment.Words"/>), where the model supports it.</summary>
    Word,
}
