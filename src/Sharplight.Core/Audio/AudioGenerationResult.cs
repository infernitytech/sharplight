namespace Sharplight.Audio;

/// <summary>The result of <see cref="IAudioGenerator.GenerateAsync"/> and <see cref="IMusicGenerator.GenerateAsync"/>.</summary>
/// <param name="Audio">The generated audio.</param>
/// <param name="Seed">The seed the noise or the sampling was drawn from (the requested one, or the one picked when the request left it unset).</param>
public sealed record AudioGenerationResult(AudioBuffer Audio, long Seed);
