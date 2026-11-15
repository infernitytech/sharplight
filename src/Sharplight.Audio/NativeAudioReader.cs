using Sharplight.Audio.Native;

namespace Sharplight.Audio;

/// <summary>Chunked reader over a native decoder handle.</summary>
internal sealed unsafe class NativeAudioReader : IAudioReader
{
    private readonly NativeDecoderHandle _handle;

    private NativeAudioReader(NativeDecoderHandle handle, AudioStreamInfo info)
    {
        _handle = handle;
        Info = info;
    }

    /// <inheritdoc />
    public AudioStreamInfo Info { get; }

    /// <summary>Opens a file.</summary>
    /// <param name="path">The file.</param>
    /// <exception cref="FileNotFoundException">The file does not exist.</exception>
    public static NativeAudioReader Open(string path)
    {
        NativeAudioLibrary.ThrowIfUnavailable();
        ArgumentException.ThrowIfNullOrEmpty(path);
        string fullPath = Path.GetFullPath(path);
        if (!File.Exists(fullPath))
        {
            throw new FileNotFoundException($"Audio file not found: {path}", fullPath);
        }

        int result = NativeMethods.sla_decoder_open_file(fullPath, out nint decoder);
        NativeException.ThrowIfFailed(result, $"Opening '{path}'");
        return Create(decoder);
    }

    /// <summary>Opens in-memory data (copied by the native library).</summary>
    /// <param name="data">The file content.</param>
    public static NativeAudioReader Open(ReadOnlyMemory<byte> data)
    {
        NativeAudioLibrary.ThrowIfUnavailable();
        if (data.IsEmpty)
        {
            throw new InvalidDataException("The audio data is empty.");
        }

        nint decoder;
        int result;
        fixed (byte* pointer = data.Span)
        {
            result = NativeMethods.sla_decoder_open_memory(pointer, (nuint)data.Length, out decoder);
        }

        NativeException.ThrowIfFailed(result, "Decoding the audio data");
        return Create(decoder);
    }

    /// <inheritdoc />
    public int Read(Span<float> destination)
    {
        int frames = destination.Length / Info.Channels;
        if (frames == 0)
        {
            return 0;
        }

        using HandleLease lease = _handle.Lease();
        ulong read;
        int result;
        fixed (float* pointer = destination)
        {
            result = NativeMethods.sla_decoder_read(lease.Value, pointer, (ulong)frames, out read);
        }

        NativeException.ThrowIfFailed(result, "Decoding audio");
        return (int)read;
    }

    /// <inheritdoc />
    public void Seek(long frame)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(frame);
        using HandleLease lease = _handle.Lease();
        int result = NativeMethods.sla_decoder_seek(lease.Value, (ulong)frame);
        if (result < 0)
        {
            throw new NotSupportedException("This audio format cannot seek to that position.");
        }
    }

    /// <inheritdoc />
    public void Dispose() => _handle.Dispose();

    private static NativeAudioReader Create(nint decoder)
    {
        var handle = new NativeDecoderHandle(decoder);
        using HandleLease lease = handle.Lease();
        NativeException.ThrowIfFailed(NativeMethods.sla_decoder_info(lease.Value, out NativeAudioInfo info), "Reading the audio format");
        string codec = info.Codec switch
        {
            1 => "wav",
            2 => "flac",
            3 => "mp3",
            4 => "vorbis",
            5 => "opus",
            _ => "unknown",
        };
        long? total = info.TotalFrames >= 0 ? info.TotalFrames : null;
        return new NativeAudioReader(handle, new AudioStreamInfo((int)info.SampleRate, (int)info.Channels, total, codec));
    }
}
