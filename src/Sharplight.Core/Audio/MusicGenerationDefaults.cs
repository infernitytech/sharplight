namespace Sharplight.Audio;

/// <summary>
/// The values a music generator uses for the options a request leaves unset (<see cref="MusicGenerationOptions"/>); get it with
/// <see cref="IMusicGenerator.Defaults"/>.
/// </summary>
/// <param name="GuidanceScale">Default classifier-free guidance scale.</param>
/// <param name="Temperature">Default sampling temperature.</param>
/// <param name="TopK">Default number of best codes the sampling keeps (0 for all).</param>
/// <param name="TopP">Default probability mass the sampling keeps (1 for all).</param>
public sealed record MusicGenerationDefaults(float GuidanceScale, float Temperature, int TopK, float TopP);
