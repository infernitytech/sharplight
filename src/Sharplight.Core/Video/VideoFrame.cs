using Sharplight.Vision;

namespace Sharplight.Video;

/// <summary>One decoded picture of a video with its presentation time.</summary>
public sealed record VideoFrame
{
    /// <summary>Zero-based position of the frame among the frames returned by the read.</summary>
    public required int Index { get; init; }

    /// <summary>The presentation time from the start of the video.</summary>
    public required TimeSpan Timestamp { get; init; }

    /// <summary>Width in pixels.</summary>
    public required int Width { get; init; }

    /// <summary>Height in pixels.</summary>
    public required int Height { get; init; }

    /// <summary>The layout of <see cref="Data"/>.</summary>
    public required VideoFrameFormat Format { get; init; }

    /// <summary>The pixels in <see cref="Format"/>; owned by the frame.</summary>
    public required byte[] Data { get; init; }

    /// <summary>The frame as an RGB image (shares <see cref="Data"/>).</summary>
    /// <returns>The image.</returns>
    /// <exception cref="InvalidOperationException">The frame is not <see cref="VideoFrameFormat.Rgb24"/>.</exception>
    public ImageInput ToImage()
    {
        if (Format != VideoFrameFormat.Rgb24)
        {
            throw new InvalidOperationException($"Only {nameof(VideoFrameFormat.Rgb24)} frames convert to an image; decode with that format.");
        }

        return new ImageInput(Width, Height, Data);
    }
}
