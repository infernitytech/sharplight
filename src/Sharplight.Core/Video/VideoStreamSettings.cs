namespace Sharplight.Video;

/// <summary>What an incremental encode (<see cref="IVideoFrameWriter"/>) writes: the frame size and rate, and where its sound comes from.</summary>
public sealed record VideoStreamSettings
{
    /// <summary>Frame width in pixels (positive).</summary>
    public required int Width { get; init; }

    /// <summary>Frame height in pixels (positive).</summary>
    public required int Height { get; init; }

    /// <summary>The playback rate in frames per second (positive, finite); <see cref="VideoEncodeOptions.FramesPerSecond"/> overrides it.</summary>
    public required double FramesPerSecond { get; init; }

    /// <summary>
    /// A media file whose first audio stream becomes the sound track, or <see langword="null"/> for a silent video. The track is copied
    /// packet by packet when the output format accepts its codec, else transcoded to the format's audio codec as the frames are written;
    /// it is never decoded into memory as a whole. It ends with the video; a file without an audio stream gives a silent video.
    /// </summary>
    public string? AudioSourcePath { get; init; }

    /// <summary>Where the sound track starts in <see cref="AudioSourcePath"/> (the time of the first written frame); zero by default.</summary>
    public TimeSpan AudioSourceStart { get; init; }

    /// <summary>Checks the values.</summary>
    /// <exception cref="ArgumentOutOfRangeException">A size or the rate is not positive, or the start is negative.</exception>
    public void Validate()
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(Width, nameof(Width));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(Height, nameof(Height));
        if (!double.IsFinite(FramesPerSecond) || FramesPerSecond <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(FramesPerSecond), FramesPerSecond, "The frame rate must be positive and finite.");
        }

        ArgumentOutOfRangeException.ThrowIfLessThan(AudioSourceStart, TimeSpan.Zero, nameof(AudioSourceStart));
    }
}
