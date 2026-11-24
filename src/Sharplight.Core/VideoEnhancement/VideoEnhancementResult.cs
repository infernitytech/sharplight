using Sharplight.Video;
using Sharplight.Vision;

namespace Sharplight.VideoEnhancement;

/// <summary>The result of <see cref="IVideoEnhancer.EnhanceAsync"/>.</summary>
/// <param name="Frames">The enhanced 8-bit RGB frames, one per input frame, in display order, all of the same size.</param>
/// <param name="FramesPerSecond">The playback rate (the request's, or 24 when it gave none).</param>
/// <param name="Seed">The seed of the noise the model started from (models without noise report 0).</param>
public sealed record VideoEnhancementResult(IReadOnlyList<ImageInput> Frames, double FramesPerSecond, long Seed)
{
    /// <summary>Width of the frames in pixels.</summary>
    public int Width => Frames[0].Width;

    /// <summary>Height of the frames in pixels.</summary>
    public int Height => Frames[0].Height;

    /// <summary>The frames as a <see cref="VideoClip"/> for the video encoders (<see cref="VideoEncoderCatalog.SaveAsync"/>).</summary>
    public VideoClip ToClip() => new(Frames, FramesPerSecond);
}
