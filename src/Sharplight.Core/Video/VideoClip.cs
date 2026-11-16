using Sharplight.Vision;

namespace Sharplight.Video;

/// <summary>A sequence of equally sized RGB frames played at a constant frame rate (for example the output of a video generation model).</summary>
public sealed record VideoClip
{
    /// <summary>Creates a clip.</summary>
    /// <param name="frames">The frames in display order; at least one, all with the size of the first and consistent pixel data.</param>
    /// <param name="framesPerSecond">The playback rate in frames per second (positive, finite).</param>
    /// <exception cref="ArgumentException">There are no frames, the frames differ in size or have the wrong byte count.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The frame rate is not positive and finite.</exception>
    public VideoClip(IReadOnlyList<ImageInput> frames, double framesPerSecond)
    {
        ArgumentNullException.ThrowIfNull(frames);
        if (frames.Count == 0)
        {
            throw new ArgumentException("A video clip needs at least one frame.", nameof(frames));
        }

        if (!double.IsFinite(framesPerSecond) || framesPerSecond <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(framesPerSecond), framesPerSecond, "The frame rate must be positive and finite.");
        }

        int width = frames[0].Width;
        int height = frames[0].Height;
        for (int i = 0; i < frames.Count; i++)
        {
            ImageInput frame = frames[i];
            ArgumentNullException.ThrowIfNull(frame);
            frame.Validate();
            if (frame.Width != width || frame.Height != height)
            {
                throw new ArgumentException($"Frame {i} is {frame.Width}x{frame.Height}, expected {width}x{height}.", nameof(frames));
            }
        }

        Frames = frames;
        FramesPerSecond = framesPerSecond;
    }

    /// <summary>The frames in display order (never empty); every frame has <see cref="Width"/> x <see cref="Height"/> pixels.</summary>
    public IReadOnlyList<ImageInput> Frames { get; }

    /// <summary>The playback rate in frames per second.</summary>
    public double FramesPerSecond { get; }

    /// <summary>Frame width in pixels (that of the first frame).</summary>
    public int Width => Frames[0].Width;

    /// <summary>Frame height in pixels (that of the first frame).</summary>
    public int Height => Frames[0].Height;

    /// <summary>The playback time of the whole clip.</summary>
    public TimeSpan Duration => TimeSpan.FromSeconds(Frames.Count / FramesPerSecond);
}
