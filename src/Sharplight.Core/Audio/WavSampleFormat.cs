namespace Sharplight.Audio;

/// <summary>Sample encodings <see cref="WavFile.Write(Stream, AudioBuffer, WavSampleFormat)"/> can produce.</summary>
public enum WavSampleFormat
{
    /// <summary>16-bit signed integer PCM (the most widely supported encoding).</summary>
    Pcm16,

    /// <summary>24-bit signed integer PCM.</summary>
    Pcm24,

    /// <summary>32-bit IEEE float (lossless for <see cref="AudioBuffer"/> samples).</summary>
    Float32,
}
