using Sharplight.Audio.Native;

namespace Sharplight.Audio;

/// <summary>
/// <see cref="IAudioEncoder"/> over the native library: WAV (16/24-bit integer, 32-bit float) and Ogg Opus. Opus supports 1 or 2 channels;
/// audio at another sample rate than 8, 12, 16, 24 or 48 kHz is resampled to 48 kHz first.
/// </summary>
public sealed unsafe class NativeAudioEncoder : IAudioEncoder
{
    private const int ChunkFrames = 65536;

    /// <inheritdoc />
    public bool SupportsFormat(AudioEncodingFormat format) => Enum.IsDefined(format);

    /// <inheritdoc />
    public byte[] Encode(AudioBuffer audio, AudioEncodeOptions? options = null)
    {
        using NativeEncoderHandle handle = Run(audio, options);
        using HandleLease lease = handle.Lease();
        NativeException.ThrowIfFailed(NativeMethods.sla_encoder_data(lease.Value, out byte* data, out nuint size), "Reading the encoded audio");
        return new ReadOnlySpan<byte>(data, checked((int)size)).ToArray();
    }

    /// <inheritdoc />
    public void Save(AudioBuffer audio, string path, AudioEncodeOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        using NativeEncoderHandle handle = Run(audio, options);
        using HandleLease lease = handle.Lease();
        NativeException.ThrowIfFailed(NativeMethods.sla_encoder_save(lease.Value, Path.GetFullPath(path)), $"Writing '{path}'");
    }

    private static NativeEncoderHandle Run(AudioBuffer audio, AudioEncodeOptions? options)
    {
        NativeAudioLibrary.ThrowIfUnavailable();
        ArgumentNullException.ThrowIfNull(audio);
        AudioEncodingFormat format = options?.Format ?? AudioEncodingFormat.WavPcm16;
        if (!Enum.IsDefined(format))
        {
            throw new NotSupportedException($"Unknown audio format {format}.");
        }

        if (format == AudioEncodingFormat.OggOpus)
        {
            if (audio.Channels > 2)
            {
                throw new NotSupportedException("Opus encoding supports mono and stereo audio.");
            }

            if (audio.SampleRate is not (8000 or 12000 or 16000 or 24000 or 48000))
            {
                audio = audio.Resample(48000);
            }
        }

        int result = NativeMethods.sla_encoder_open((int)format + 1, (uint)audio.SampleRate, (uint)audio.Channels, options?.BitrateBitsPerSecond ?? 0, out nint encoder);
        if (result == (int)NativeResult.Unsupported)
        {
            throw new NotSupportedException($"{format} does not support {audio.SampleRate} Hz with {audio.Channels} channel(s).");
        }

        NativeException.ThrowIfFailed(result, "Creating the encoder");
        var handle = new NativeEncoderHandle(encoder);
        try
        {
            using HandleLease lease = handle.Lease();
            int channels = audio.Channels;
            int frames = audio.FrameCount;
            for (int offset = 0; offset < frames; offset += ChunkFrames)
            {
                int count = Math.Min(ChunkFrames, frames - offset);
                fixed (float* pointer = &audio.Samples[offset * channels])
                {
                    NativeException.ThrowIfFailed(NativeMethods.sla_encoder_write(lease.Value, pointer, (ulong)count), "Encoding audio");
                }
            }

            NativeException.ThrowIfFailed(NativeMethods.sla_encoder_finish(lease.Value), "Finishing the encoded audio");
            return handle;
        }
        catch
        {
            handle.Dispose();
            throw;
        }
    }
}
