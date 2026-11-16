namespace Sharplight.Audio;

/// <summary>Settings for <see cref="IAudioEncoder"/>.</summary>
public sealed class AudioEncodeOptions
{
    /// <summary>The output format. Default: 16-bit WAV.</summary>
    public AudioEncodingFormat Format { get; set; } = AudioEncodingFormat.WavPcm16;

    /// <summary>
    /// Target bit rate in bits per second for lossy formats; zero or negative selects the encoder's default (64 kbit/s per channel for Opus).
    /// Ignored by lossless formats.
    /// </summary>
    public int BitrateBitsPerSecond { get; set; }
}
