namespace Sharplight.Transcribe;

/// <summary>How the transcript is written.</summary>
public enum TranscriptFormat
{
    /// <summary>The plain text.</summary>
    Text,

    /// <summary>One line per segment with its start and end time.</summary>
    Segments,

    /// <summary>SubRip subtitles.</summary>
    Srt,

    /// <summary>JSON: text, language, duration and the segments.</summary>
    Json,
}
