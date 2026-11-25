using Sharplight.Video;
using Sharplight.Vision;

namespace Sharplight.VideoEnhancement;

/// <summary>The input of <see cref="IVideoEnhancer"/>: the frames of a clip, or a single image.</summary>
public sealed record VideoEnhancementRequest
{
    /// <summary>The frames in display order; at least one, all of the same size. One frame enhances a still image.</summary>
    public required IReadOnlyList<ImageInput> Frames { get; init; }

    /// <summary>The playback rate of the clip, passed through to the result; <see langword="null"/> when unknown (the result then reports 24).</summary>
    public double? FramesPerSecond { get; init; }

    /// <summary>A request for the frames of <paramref name="clip"/> at its rate.</summary>
    /// <param name="clip">A decoded clip (e.g. from <see cref="VideoDecoderCatalog"/>).</param>
    public static VideoEnhancementRequest FromClip(VideoClip clip)
    {
        ArgumentNullException.ThrowIfNull(clip);
        return new VideoEnhancementRequest { Frames = clip.Frames, FramesPerSecond = clip.FramesPerSecond };
    }

    /// <summary>A request for one still image.</summary>
    /// <param name="image">The image.</param>
    public static VideoEnhancementRequest FromImage(ImageInput image)
    {
        ArgumentNullException.ThrowIfNull(image);
        return new VideoEnhancementRequest { Frames = [image] };
    }
}
