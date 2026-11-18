namespace Sharplight.Architectures;

/// <summary>One segment of a VibeVoice-ASR transcript: the model writes a JSON array of <c>{"Start":…,"End":…,"Speaker":…,"Content":"…"}</c> objects.</summary>
/// <param name="Start">The start time in seconds from the start of the audio the model heard.</param>
/// <param name="End">The end time in seconds.</param>
/// <param name="Speaker">The speaker number the model assigned.</param>
/// <param name="Content">The words.</param>
internal sealed record VibeVoiceSegment(double Start, double End, int Speaker, string Content);
