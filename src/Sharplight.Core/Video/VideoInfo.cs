namespace Sharplight.Video;

/// <summary>What a video file contains (<see cref="IVideoDecoder.ProbeAsync"/>).</summary>
public sealed record VideoInfo
{
    /// <summary>The container format as the decoder names it.</summary>
    public required string Container { get; init; }

    /// <summary>The video codec as the decoder names it.</summary>
    public required string VideoCodec { get; init; }

    /// <summary>Width in pixels (after the stored rotation is ignored).</summary>
    public required int Width { get; init; }

    /// <summary>Height in pixels.</summary>
    public required int Height { get; init; }

    /// <summary>The average frame rate, or 0 when the file does not tell.</summary>
    public required double FramesPerSecond { get; init; }

    /// <summary>The playback time, or <see cref="TimeSpan.Zero"/> when unknown.</summary>
    public required TimeSpan Duration { get; init; }

    /// <summary>The number of video frames, stored or estimated from the duration and frame rate; 0 when unknown.</summary>
    public required long FrameCount { get; init; }

    /// <summary>The audio codec as the decoder names it, or <see langword="null"/> without an audio track.</summary>
    public string? AudioCodec { get; init; }

    /// <summary>The sample rate of the audio track in Hz, or 0 without one.</summary>
    public int AudioSampleRate { get; init; }

    /// <summary>The channel count of the audio track, or 0 without one.</summary>
    public int AudioChannels { get; init; }
}
