namespace Sharplight.Speech;

/// <summary>The result of recognizing a whole recording.</summary>
/// <param name="Text">The segment texts concatenated.</param>
/// <param name="Language">The language given or detected first (ISO 639-1), or <see langword="null"/> when unknown.</param>
/// <param name="Segments">The segments in time order.</param>
/// <param name="Duration">Length of the recognized audio.</param>
public sealed record Transcription(string Text, string? Language, IReadOnlyList<TranscriptionSegment> Segments, TimeSpan Duration);
