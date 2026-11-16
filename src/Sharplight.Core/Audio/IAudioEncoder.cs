namespace Sharplight.Audio;

/// <summary>Encodes <see cref="AudioBuffer"/>s to audio files.</summary>
public interface IAudioEncoder
{
    /// <summary>Whether <paramref name="format"/> can be written by this encoder.</summary>
    /// <param name="format">The format to test.</param>
    public bool SupportsFormat(AudioEncodingFormat format);

    /// <summary>Encodes audio to a byte array.</summary>
    /// <param name="audio">The audio.</param>
    /// <param name="options">Format and bit rate; <see langword="null"/> selects 16-bit WAV.</param>
    /// <exception cref="NotSupportedException">The format or the audio's sample rate / channel count is not supported by the format.</exception>
    public byte[] Encode(AudioBuffer audio, AudioEncodeOptions? options = null);

    /// <summary>Encodes audio to a file, replacing it.</summary>
    /// <param name="audio">The audio.</param>
    /// <param name="path">The file to write.</param>
    /// <param name="options">Format and bit rate; <see langword="null"/> selects 16-bit WAV.</param>
    /// <exception cref="NotSupportedException">The format or the audio's sample rate / channel count is not supported by the format.</exception>
    public void Save(AudioBuffer audio, string path, AudioEncodeOptions? options = null);
}
