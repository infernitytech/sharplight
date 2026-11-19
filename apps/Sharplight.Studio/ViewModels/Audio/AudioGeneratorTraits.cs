using Sharplight.Audio;

namespace Sharplight.Studio.ViewModels.Audio;

/// <summary>What the Audio page offers for a diffusion audio generator (kind <c>audio-generation</c>), from what its client says it can do.</summary>
/// <param name="Lyrics">The model sings lyrics in a chosen language (ACE-Step).</param>
/// <param name="NegativePrompt">The model takes a negative prompt (Stable Audio Open).</param>
/// <param name="Guidance">The model runs with classifier-free guidance (not a distilled model).</param>
/// <param name="Shift">The model's timestep schedule can be shifted.</param>
/// <param name="DefaultGuidance">The model's own guidance scale, or <see langword="null"/>.</param>
internal sealed record AudioGeneratorTraits(bool Lyrics, bool NegativePrompt, bool Guidance, bool Shift, float? DefaultGuidance)
{
    /// <summary>The traits of a client.</summary>
    /// <param name="client">The leased generator.</param>
    public static AudioGeneratorTraits For(IAudioGenerator client)
    {
        ArgumentNullException.ThrowIfNull(client);
        return new AudioGeneratorTraits(client.SupportsLyrics, client.SupportsNegativePrompt, client.DefaultGuidanceScale is not null, client.SupportsShift,
            client.DefaultGuidanceScale);
    }
}
