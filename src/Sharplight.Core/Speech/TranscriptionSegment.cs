namespace Sharplight.Speech;

/// <summary>A contiguous piece of recognized speech.</summary>
public sealed record TranscriptionSegment
{
    /// <summary>Start time from the beginning of the audio.</summary>
    public required TimeSpan Start { get; init; }

    /// <summary>End time from the beginning of the audio.</summary>
    public required TimeSpan End { get; init; }

    /// <summary>The recognized text (or its translation, see <see cref="SpeechRecognitionOptions.Translate"/>).</summary>
    public required string Text { get; init; }

    /// <summary>Language of the segment (ISO 639-1, e.g. <c>en</c>), given or detected; <see langword="null"/> when unknown.</summary>
    public string? Language { get; init; }

    /// <summary>Per-word timings when <see cref="TimestampGranularity.Word"/> was requested and supported; otherwise empty.</summary>
    public IReadOnlyList<WordTiming> Words { get; init; } = [];

    /// <summary>Mean log probability of the segment's tokens, or <see langword="null"/> when not available.</summary>
    public double? AverageLogProbability { get; init; }

    /// <summary>The model's probability that the window holds no speech, or <see langword="null"/> when not available.</summary>
    public double? NoSpeechProbability { get; init; }
}
