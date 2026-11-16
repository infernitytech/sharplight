namespace Sharplight.Audio;

/// <summary>
/// A playback stream: the application writes samples, the device plays them. One thread writes; <see cref="Start"/> and
/// <see cref="Stop"/> may come from any thread. Dispose to close the device.
/// </summary>
public interface IAudioOutput : IDisposable
{
    /// <summary>Sample rate of the samples passed to <see cref="Write"/>.</summary>
    public int SampleRate { get; }

    /// <summary>Channels per frame of the samples passed to <see cref="Write"/>.</summary>
    public int Channels { get; }

    /// <summary>Frames that <see cref="Write"/> accepts right now.</summary>
    public int FreeFrames { get; }

    /// <summary>Frames written but not yet played.</summary>
    public int QueuedFrames { get; }

    /// <summary>Transfer counters.</summary>
    public AudioStreamStatistics Statistics { get; }

    /// <summary>Starts the device; it plays silence until samples are written.</summary>
    public void Start();

    /// <summary>Stops the device; queued samples stay queued.</summary>
    public void Stop();

    /// <summary>Queues samples without blocking.</summary>
    /// <param name="samples">Interleaved float samples (whole frames).</param>
    /// <returns>The number of frames accepted, less than offered when the buffer is full.</returns>
    public int Write(ReadOnlySpan<float> samples);
}
