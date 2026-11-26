using System.Runtime.CompilerServices;
using Sharplight.Video;
using Sharplight.Vision;

namespace Sharplight.VideoEnhancement;

/// <summary>
/// The input of <see cref="IVideoEnhancer"/>: the frames of a clip, or a single image. The frames come either as a list in memory
/// (<see cref="Frames"/>) or from a source that is read while the enhancement runs (<see cref="FrameSource"/>, for example a video decoder),
/// so a clip of any length is enhanced with the memory of a few batches.
/// </summary>
public sealed record VideoEnhancementRequest
{
    /// <summary>
    /// The frames in display order, all of the same size; one frame enhances a still image. Exactly one of <see cref="Frames"/> and
    /// <see cref="FrameSource"/> is set.
    /// </summary>
    public IReadOnlyList<ImageInput>? Frames { get; init; }

    /// <summary>
    /// The frames in display order, enumerated once and only as far as the enhancement has got (an enhancer holds the frames of the batches
    /// it is working on, never the clip); all of the same size, at least one. Exactly one of <see cref="Frames"/> and
    /// <see cref="FrameSource"/> is set.
    /// </summary>
    public IAsyncEnumerable<ImageInput>? FrameSource { get; init; }

    /// <summary>
    /// How many frames <see cref="FrameSource"/> will deliver, when known. An estimate is fine: it sizes the batches of a whole-clip request
    /// and the progress totals, never the output. <see langword="null"/> when unknown; ignored with <see cref="Frames"/>.
    /// </summary>
    public long? FrameCount { get; init; }

    /// <summary>The playback rate of the clip, passed through to the result; <see langword="null"/> when unknown (the result then reports 24).</summary>
    public double? FramesPerSecond { get; init; }

    /// <summary>The number of frames when known: the count of <see cref="Frames"/>, else a positive <see cref="FrameCount"/>.</summary>
    public long? KnownFrameCount => Frames is { } frames ? frames.Count : FrameCount is > 0 and var count ? count : null;

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

    /// <summary>A request that reads its frames from <paramref name="frames"/> while it runs (for example the frames a decoder streams).</summary>
    /// <param name="frames">The frames in display order.</param>
    /// <param name="framesPerSecond">The playback rate, or <see langword="null"/> when unknown.</param>
    /// <param name="frameCount">How many frames the source will deliver, or <see langword="null"/> when unknown (see <see cref="FrameCount"/>).</param>
    public static VideoEnhancementRequest FromSource(IAsyncEnumerable<ImageInput> frames, double? framesPerSecond = null, long? frameCount = null)
    {
        ArgumentNullException.ThrowIfNull(frames);
        return new VideoEnhancementRequest { FrameSource = frames, FramesPerSecond = framesPerSecond, FrameCount = frameCount };
    }

    /// <summary>The frames of the request in order, from <see cref="Frames"/> or <see cref="FrameSource"/>.</summary>
    /// <param name="cancellationToken">Stops the reading.</param>
    /// <exception cref="ArgumentException">Neither or both of <see cref="Frames"/> and <see cref="FrameSource"/> are set.</exception>
    public IAsyncEnumerable<ImageInput> ReadFramesAsync(CancellationToken cancellationToken = default)
    {
        if ((Frames is null) == (FrameSource is null))
        {
            throw new ArgumentException($"A video-enhancement request needs exactly one of {nameof(Frames)} and {nameof(FrameSource)}.");
        }

        return FrameSource is { } source ? FromSource(source, cancellationToken) : FromList(Frames!, cancellationToken);
    }

    private static async IAsyncEnumerable<ImageInput> FromSource(IAsyncEnumerable<ImageInput> source, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        await foreach (var frame in source.WithCancellation(cancellationToken).ConfigureAwait(false))
        {
            yield return frame;
        }
    }

    private static async IAsyncEnumerable<ImageInput> FromList(IReadOnlyList<ImageInput> frames, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        await Task.CompletedTask.ConfigureAwait(false);
        for (int i = 0; i < frames.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            yield return frames[i];
        }
    }
}
