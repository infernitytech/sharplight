using Sharplight.Audio.Native;

namespace Sharplight.Audio;

/// <summary>Capture stream over a native stream handle.</summary>
internal sealed unsafe class NativeAudioInput : IAudioInput
{
    private readonly NativeContextHandle _context;
    private readonly NativeStreamHandle _stream;
    private bool _disposed;

    /// <summary>Wraps an opened stream; takes over one reference on <paramref name="context"/>.</summary>
    /// <param name="context">The device context the stream belongs to (released on dispose).</param>
    /// <param name="stream">The stream.</param>
    /// <param name="sampleRate">Sample rate of exchanged samples.</param>
    /// <param name="channels">Channels per frame.</param>
    public NativeAudioInput(NativeContextHandle context, NativeStreamHandle stream, int sampleRate, int channels)
    {
        _context = context;
        _stream = stream;
        SampleRate = sampleRate;
        Channels = channels;
    }

    /// <inheritdoc />
    public int SampleRate { get; }

    /// <inheritdoc />
    public int Channels { get; }

    /// <inheritdoc />
    public int AvailableFrames
    {
        get
        {
            using HandleLease lease = _stream.Lease();
            NativeException.ThrowIfFailed(NativeMethods.sla_stream_available(lease.Value, out uint frames), "Querying the audio buffer");
            return (int)frames;
        }
    }

    /// <inheritdoc />
    public AudioStreamStatistics Statistics => NativeAudioStatistics.Read(_stream);

    /// <inheritdoc />
    public void Start()
    {
        using HandleLease lease = _stream.Lease();
        NativeException.ThrowIfFailed(NativeMethods.sla_stream_start(lease.Value), "Starting capture");
    }

    /// <inheritdoc />
    public void Stop()
    {
        using HandleLease lease = _stream.Lease();
        NativeException.ThrowIfFailed(NativeMethods.sla_stream_stop(lease.Value), "Stopping capture");
    }

    /// <inheritdoc />
    public int Read(Span<float> destination)
    {
        int frames = destination.Length / Channels;
        if (frames == 0)
        {
            return 0;
        }

        using HandleLease lease = _stream.Lease();
        uint read;
        int result;
        fixed (float* pointer = destination)
        {
            result = NativeMethods.sla_stream_read(lease.Value, pointer, (uint)frames, out read);
        }

        NativeException.ThrowIfFailed(result, "Reading captured audio");
        return (int)read;
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
