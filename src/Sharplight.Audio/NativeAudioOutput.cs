using Sharplight.Audio.Native;

namespace Sharplight.Audio;

/// <summary>Playback stream over a native stream handle.</summary>
internal sealed unsafe class NativeAudioOutput : IAudioOutput
{
    private readonly NativeContextHandle _context;
    private readonly NativeStreamHandle _stream;
    private readonly int _ringFrames;
    private bool _disposed;

    /// <summary>Wraps an opened stream; takes over one reference on <paramref name="context"/>.</summary>
    /// <param name="context">The device context the stream belongs to (released on dispose).</param>
    /// <param name="stream">The stream.</param>
    /// <param name="sampleRate">Sample rate of exchanged samples.</param>
    /// <param name="channels">Channels per frame.</param>
    /// <param name="ringFrames">Ring buffer capacity in frames.</param>
    public NativeAudioOutput(NativeContextHandle context, NativeStreamHandle stream, int sampleRate, int channels, int ringFrames)
    {
        _context = context;
        _stream = stream;
        SampleRate = sampleRate;
        Channels = channels;
        _ringFrames = ringFrames;
    }

    /// <inheritdoc />
    public int SampleRate { get; }

    /// <inheritdoc />
    public int Channels { get; }

    /// <inheritdoc />
    public int FreeFrames
    {
        get
        {
            using HandleLease lease = _stream.Lease();
            NativeException.ThrowIfFailed(NativeMethods.sla_stream_available(lease.Value, out uint frames), "Querying the audio buffer");
            return (int)frames;
        }
    }

    /// <inheritdoc />
    public int QueuedFrames => _ringFrames - FreeFrames;

    /// <inheritdoc />
    public AudioStreamStatistics Statistics => NativeAudioStatistics.Read(_stream);

    /// <inheritdoc />
    public void Start()
    {
        using HandleLease lease = _stream.Lease();
        NativeException.ThrowIfFailed(NativeMethods.sla_stream_start(lease.Value), "Starting playback");
    }

    /// <inheritdoc />
    public void Stop()
    {
        using HandleLease lease = _stream.Lease();
        NativeException.ThrowIfFailed(NativeMethods.sla_stream_stop(lease.Value), "Stopping playback");
    }

    /// <inheritdoc />
    public int Write(ReadOnlySpan<float> samples)
    {
        int frames = samples.Length / Channels;
        if (frames == 0)
        {
            return 0;
        }

        using HandleLease lease = _stream.Lease();
        uint written;
        int result;
        fixed (float* pointer = samples)
        {
            result = NativeMethods.sla_stream_write(lease.Value, pointer, (uint)frames, out written);
        }

        NativeException.ThrowIfFailed(result, "Writing audio");
        return (int)written;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _stream.Dispose();
        _context.DangerousRelease();
    }
}
