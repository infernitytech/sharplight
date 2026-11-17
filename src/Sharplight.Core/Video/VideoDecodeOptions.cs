namespace Sharplight.Video;

/// <summary>What part of a video to decode and in which form.</summary>
public sealed record VideoDecodeOptions
{
    /// <summary>The pixel layout of the returned frames.</summary>
    public VideoFrameFormat Format { get; init; } = VideoFrameFormat.Rgb24;

    /// <summary>Where to start; frames before this time are skipped (the decoder seeks to the key frame before it).</summary>
    public TimeSpan Start { get; init; }

    /// <summary>How much to decode from <see cref="Start"/>; <see langword="null"/> reads to the end.</summary>
    public TimeSpan? Duration { get; init; }

    /// <summary>The most frames to return; <see langword="null"/> for no limit.</summary>
    public int? MaxFrames { get; init; }

    /// <summary>
    /// Resample to this frame rate (positive) by dropping or repeating frames; <see langword="null"/> returns every decoded frame at the
    /// source's timing.
    /// </summary>
    public double? FramesPerSecond { get; init; }

    /// <summary>Scale the frames to this width in pixels; with only one of <see cref="Width"/> / <see cref="Height"/> the aspect ratio is kept.</summary>
    public int? Width { get; init; }

    /// <summary>Scale the frames to this height in pixels; with only one of <see cref="Width"/> / <see cref="Height"/> the aspect ratio is kept.</summary>
    public int? Height { get; init; }
}
