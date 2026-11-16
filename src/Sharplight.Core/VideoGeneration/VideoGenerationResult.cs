using Sharplight.Vision;

namespace Sharplight.VideoGeneration;

/// <summary>The result of <see cref="IVideoGenerator.GenerateAsync"/>.</summary>
/// <param name="Frames">The generated 8-bit RGB frames, in display order, all of the same size.</param>
/// <param name="FramesPerSecond">The playback rate.</param>
/// <param name="Seed">The seed of the initial noise.</param>
public sealed record VideoGenerationResult(IReadOnlyList<ImageInput> Frames, double FramesPerSecond, long Seed)
{
    /// <summary>Duration of the video.</summary>
    public TimeSpan Duration => TimeSpan.FromSeconds(Frames.Count / FramesPerSecond);
}
