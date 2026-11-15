namespace Sharplight.Speech;

/// <summary>The decoded tokens of one audio window, before they are split into segments.</summary>
/// <param name="Tokens">Text and timestamp tokens after the prompt, without the end token.</param>
/// <param name="AverageLogProbability">Mean log probability of the chosen tokens including the end token.</param>
/// <param name="NoSpeechProbability">The no-speech probability of the window, or null when the model has no such token.</param>
internal sealed record WindowDecoding(IReadOnlyList<int> Tokens, double AverageLogProbability, double? NoSpeechProbability);
