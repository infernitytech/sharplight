using Sharplight.Audio;

namespace Sharplight.Speech;

/// <summary>The result of <see cref="ISpeechSynthesizer.SynthesizeAsync"/>.</summary>
/// <param name="Audio">The synthesized audio (mono unless the model produces more channels).</param>
/// <param name="Voice">The voice used, or <see langword="null"/> for single-voice models.</param>
public sealed record SpeechSynthesisResult(AudioBuffer Audio, string? Voice);
