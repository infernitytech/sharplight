namespace Sharplight.Audio;

/// <summary>The built-in <see cref="IAudioEncoder"/>: WAV only (<see cref="WavFile"/>). Used when no native encoder is registered.</summary>
public sealed class WavAudioEncoder : IAudioEncoder
{
    /// <inheritdoc />
    public bool SupportsFormat(AudioEncodingFormat format) => format != AudioEncodingFormat.OggOpus;

    /// <inheritdoc />
    public byte[] Encode(AudioBuffer audio, AudioEncodeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(audio);
        using var stream = new MemoryStream();
        WavFile.Write(stream, audio, ToWavFormat(options));
        return stream.ToArray();
    }

    /// <inheritdoc />
    public void Save(AudioBuffer audio, string path, AudioEncodeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(audio);
        WavFile.Write(path, audio, ToWavFormat(options));
    }

    private static WavSampleFormat ToWavFormat(AudioEncodeOptions? options) => (options?.Format ?? AudioEncodingFormat.WavPcm16) switch
    {
        AudioEncodingFormat.WavPcm16 => WavSampleFormat.Pcm16,
        AudioEncodingFormat.WavPcm24 => WavSampleFormat.Pcm24,
        AudioEncodingFormat.WavFloat32 => WavSampleFormat.Float32,
        var other => throw new NotSupportedException($"The built-in encoder cannot write {other}; register Sharplight.Audio (services.AddSharplightAudio())."),
    };
}
