namespace Sharplight.Audio;

/// <summary>
/// <see cref="IAudioDecoder"/> over the native library: WAV (16/24/32-bit integer, float), FLAC, MP3, Ogg Vorbis and Ogg Opus, detected from
/// the content. Opus is always decoded at 48 kHz (the codec's native rate).
/// </summary>
public sealed class NativeAudioDecoder : IAudioDecoder
{
    private const int ChunkFrames = 16384;

    /// <inheritdoc />
    public AudioBuffer Decode(string path)
    {
        using NativeAudioReader reader = NativeAudioReader.Open(path);
        return ReadAll(reader);
    }

    /// <inheritdoc />
    public AudioBuffer Decode(ReadOnlyMemory<byte> data)
    {
        using NativeAudioReader reader = NativeAudioReader.Open(data);
        return ReadAll(reader);
    }

    /// <inheritdoc />
    public IAudioReader OpenReader(string path) => NativeAudioReader.Open(path);

    /// <inheritdoc />
    public IAudioReader OpenReader(ReadOnlyMemory<byte> data) => NativeAudioReader.Open(data);

    private static AudioBuffer ReadAll(NativeAudioReader reader)
    {
        AudioStreamInfo info = reader.Info;
        int channels = info.Channels;
        long expected = info.TotalFrames ?? 0;
        var samples = new float[(int)Math.Min(Math.Max(expected, ChunkFrames), int.MaxValue / channels - ChunkFrames) * channels];
        int frames = 0;
        while (true)
        {
            if ((frames + ChunkFrames) * channels > samples.Length)
            {
                Array.Resize(ref samples, Math.Min(samples.Length * 2, int.MaxValue / 2 / channels * channels));
            }

            int read = reader.Read(samples.AsSpan(frames * channels, ChunkFrames * channels));
            if (read == 0)
            {
                break;
            }

            frames += read;
        }

        if (frames * channels != samples.Length)
        {
            Array.Resize(ref samples, frames * channels);
        }

        return new AudioBuffer(samples, info.SampleRate, channels);
    }
}
