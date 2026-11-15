namespace Sharplight.Audio;

/// <summary>
/// Decodes audio files to <see cref="AudioBuffer"/>s. The format is detected from the content, not the extension.
/// The built-in implementation reads WAV only; <c>Sharplight.Audio</c> adds FLAC, MP3, Ogg Vorbis and Ogg Opus.
/// </summary>
public interface IAudioDecoder
{
    /// <summary>Decodes a whole file at its stored sample rate and channel count.</summary>
    /// <param name="path">The file to read.</param>
    /// <exception cref="FileNotFoundException">The file does not exist.</exception>
    /// <exception cref="InvalidDataException">The file is corrupt or in an unsupported format.</exception>
    public AudioBuffer Decode(string path);

    /// <summary>Decodes a whole in-memory file.</summary>
    /// <param name="data">The file content.</param>
    /// <exception cref="InvalidDataException">The data is corrupt or in an unsupported format.</exception>
    public AudioBuffer Decode(ReadOnlyMemory<byte> data);

    /// <summary>Opens a file for chunked reading.</summary>
    /// <param name="path">The file to read.</param>
    /// <exception cref="FileNotFoundException">The file does not exist.</exception>
    /// <exception cref="InvalidDataException">The file is corrupt or in an unsupported format.</exception>
    public IAudioReader OpenReader(string path);

    /// <summary>Opens in-memory data for chunked reading; the data is copied.</summary>
    /// <param name="data">The file content.</param>
    /// <exception cref="InvalidDataException">The data is corrupt or in an unsupported format.</exception>
    public IAudioReader OpenReader(ReadOnlyMemory<byte> data);
}
