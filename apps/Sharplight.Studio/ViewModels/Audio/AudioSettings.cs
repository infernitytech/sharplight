using Sharplight.Audio;

namespace Sharplight.Studio.ViewModels.Audio;

/// <summary>The settings of one run of the Audio page.</summary>
/// <param name="Codec">The codec request.</param>
/// <param name="VoiceActivity">The voice activity request.</param>
/// <param name="Tagging">The zero-shot tagging request.</param>
/// <param name="Separation">The separation request.</param>
/// <param name="Music">The music request without its prompt (the prompt is the page's text).</param>
internal sealed record AudioSettings(AudioCodecOptions Codec, VoiceActivityOptions VoiceActivity, AudioClassificationOptions Tagging, AudioSeparationOptions Separation, MusicGenerationRequest Music);
