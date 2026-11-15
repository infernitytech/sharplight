namespace Sharplight.Audio;

/// <summary>Output formats an <see cref="IAudioEncoder"/> can write.</summary>
public enum AudioEncodingFormat
{
    /// <summary>WAV with 16-bit signed integer PCM samples.</summary>
    WavPcm16,

    /// <summary>WAV with 24-bit signed integer PCM samples.</summary>
    WavPcm24,

    /// <summary>WAV with 32-bit IEEE float samples (lossless for <see cref="AudioBuffer"/> samples).</summary>
    WavFloat32,

    /// <summary>Opus in an Ogg container (lossy; needs an encoder that supports it, see <see cref="IAudioEncoder.SupportsFormat"/>).</summary>
    OggOpus,
}
