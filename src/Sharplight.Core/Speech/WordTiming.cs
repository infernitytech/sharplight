namespace Sharplight.Speech;

/// <summary>One recognized word with its position in the audio.</summary>
/// <param name="Text">The word as it appears in the segment text (including leading whitespace, if any).</param>
/// <param name="Start">Start time from the beginning of the audio.</param>
/// <param name="End">End time from the beginning of the audio.</param>
/// <param name="Probability">The model's confidence in the word, in [0, 1], or <see langword="null"/> when not available.</param>
public sealed record WordTiming(string Text, TimeSpan Start, TimeSpan End, float? Probability = null);
