namespace Sharplight.Audio;

/// <summary>
/// The values an audio generator uses for the options a request leaves unset (<see cref="AudioGenerationOptions"/>) and what it supports; get it with
/// <see cref="IAudioGenerator.Defaults"/>.
/// </summary>
/// <param name="Steps">Default number of denoising steps.</param>
/// <param name="GuidanceScale">Default classifier-free guidance scale, or <see langword="null"/> when the model runs without guidance (a distilled model).</param>
/// <param name="SupportsLyrics">The model sings lyrics (<see cref="AudioGenerationRequest.Lyrics"/> and <see cref="AudioGenerationRequest.Language"/>); other models ignore them.</param>
/// <param name="SupportsNegativePrompt">The model takes a negative prompt (<see cref="AudioGenerationRequest.NegativePrompt"/>); other models ignore it.</param>
/// <param name="SupportsShift">The model's sampler shifts its timestep schedule (<see cref="AudioGenerationOptions.Shift"/>); other models ignore the shift.</param>
public sealed record AudioGenerationDefaults(int Steps, float? GuidanceScale, bool SupportsLyrics = false, bool SupportsNegativePrompt = false, bool SupportsShift = false);
