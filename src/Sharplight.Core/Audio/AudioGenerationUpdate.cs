namespace Sharplight.Audio;

/// <summary>Progress of <see cref="IAudioGenerator.GenerateStreamingAsync"/> and <see cref="IMusicGenerator.GenerateStreamingAsync"/>.</summary>
/// <param name="Step">Units completed: denoising steps of a diffusion model, generated code frames of an autoregressive one.</param>
/// <param name="TotalSteps">Units of the request.</param>
/// <param name="Result">The finished audio on the last update; <see langword="null"/> before.</param>
/// <param name="Audio">
/// Audio decoded while the request is still running (an autoregressive model's next piece, decoded in windows with some context on both sides, so the
/// concatenated pieces differ slightly from <see cref="Result"/> at the window borders); otherwise <see langword="null"/>.
/// </param>
public sealed record AudioGenerationUpdate(int Step, int TotalSteps, AudioGenerationResult? Result = null, AudioBuffer? Audio = null);
