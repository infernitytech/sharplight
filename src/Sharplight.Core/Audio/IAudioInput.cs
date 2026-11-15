namespace Sharplight.Audio;

/// <summary>
/// A capture stream: the device delivers samples, the application reads them. One thread reads; <see cref="Start"/> and
/// <see cref="Stop"/> may come from any thread. Dispose to close the device.
/// </summary>
public interface IAudioInput : IDisposable
{
    /// <summary>Sample rate of the samples returned by <see cref="Read"/>.</summary>
    public int SampleRate { get; }

    /// <summary>Channels per frame of the samples returned by <see cref="Read"/>.</summary>
    public int Channels { get; }

    /// <summary>Frames <see cref="Read"/> can return right now.</summary>
    public int AvailableFrames { get; }

    /// <summary>Transfer counters.</summary>
    public AudioStreamStatistics Statistics { get; }

    /// <summary>Starts capturing.</summary>
    public void Start();

    /// <summary>Stops capturing; unread samples stay available.</summary>
    public void Stop();

    /// <summary>Copies captured samples without blocking.</summary>
    /// <param name="destination">Receives interleaved float samples.</param>
    /// <returns>The number of frames copied (zero when nothing is buffered).</returns>
    public int Read(Span<float> destination);
}
