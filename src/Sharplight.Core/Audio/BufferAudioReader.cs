namespace Sharplight.Audio;

/// <summary>Serves an already decoded <see cref="AudioBuffer"/> through <see cref="IAudioReader"/>.</summary>
internal sealed class BufferAudioReader : IAudioReader
{
    private readonly AudioBuffer _buffer;
    private long _position;

    /// <summary>Creates a reader over the buffer.</summary>
    /// <param name="buffer">The decoded audio.</param>
    /// <param name="codec">Codec name reported in <see cref="Info"/>.</param>
    public BufferAudioReader(AudioBuffer buffer, string codec)
    {
        _buffer = buffer;
        Info = new AudioStreamInfo(buffer.SampleRate, buffer.Channels, buffer.FrameCount, codec);
    }

    /// <inheritdoc />
    public AudioStreamInfo Info { get; }

    /// <inheritdoc />
    public int Read(Span<float> destination)
    {
        long remaining = _buffer.FrameCount - _position;
        int frames = (int)Math.Min(remaining, destination.Length / _buffer.Channels);
        if (frames <= 0)
        {
            return 0;
        }

        _buffer.Samples.AsSpan((int)(_position * _buffer.Channels), frames * _buffer.Channels).CopyTo(destination);
        _position += frames;
        return frames;
    }

    /// <inheritdoc />
    public void Seek(long frame)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(frame);
        _position = Math.Min(frame, _buffer.FrameCount);
    }

    /// <inheritdoc />
    public void Dispose()
    {
    }
}
