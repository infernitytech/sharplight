using Sharplight.Audio;

namespace Sharplight.Studio.ViewModels.Audio;

/// <summary>The settings of one run of the Audio page.</summary>
/// <param name="Codec">The codec request.</param>
/// <param name="VoiceActivity">The voice activity request.</param>
internal sealed record AudioSettings(AudioCodecOptions Codec, VoiceActivityOptions VoiceActivity);
