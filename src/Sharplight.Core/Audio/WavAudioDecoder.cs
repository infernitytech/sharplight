namespace Sharplight.Audio;

/// <summary>The built-in <see cref="IAudioDecoder"/>: WAV only (<see cref="WavFile"/>). Used when no native decoder is registered.</summary>
public sealed class WavAudioDecoder : IAudioDecoder
{
    /// <inheritdoc />
    public AudioBuffer Decode(string path) => WavFile.Read(path);

    /// <inheritdoc />
    public AudioBuffer Decode(ReadOnlyMemory<byte> data)
    {
        using var stream = new MemoryStream(data.ToArray(), writable: false);
        return WavFile.Read(stream);
    }

    /// <inheritdoc />
    public IAudioReader OpenReader(string path) => new BufferAudioReader(Decode(path), "wav");

    /// <inheritdoc />
    public IAudioReader OpenReader(ReadOnlyMemory<byte> data) => new BufferAudioReader(Decode(data), "wav");
}
