using Sharplight.Tasks;
using Sharplight.Vision;

namespace Sharplight.VideoEnhancement;

/// <summary>
/// Video enhancement (kind <see cref="Models.ModelKinds.VideoEnhancement"/>): the frames of a clip (or one image) in, restored and usually
/// enlarged frames out (video super-resolution and restoration). Follows the task-client pattern of every model kind
/// (<see cref="ITaskClient"/>) and the shapes of <see cref="VideoGeneration.IVideoGenerator"/>. Implementations come from
/// <see cref="ITaskClientFactory"/> services of the architecture packages. Not thread-safe: run one request at a time. The work runs on a
/// background thread, never on the caller's thread.
/// </summary>
/// <remarks>
/// <see cref="EnhanceStreamingAsync"/> is the one path: it reads the request's frames batch by batch (from a list or a
/// <see cref="VideoEnhancementRequest.FrameSource"/>), finishes each batch before reading far ahead and hands the finished frames out as it
/// goes, so its memory is that of a few batches whatever the clip's length. <see cref="EnhanceAsync"/> collects that stream into a result.
/// </remarks>
public interface IVideoEnhancer : ITaskClient
{
    /// <summary>
    /// The model's own defaults (native size, steps, guidance, sampler and what it supports), which the <see langword="null"/> values of the
    /// per-request options resolve to.
    /// </summary>
    public VideoEnhancementDefaults Defaults { get; }

    /// <summary>Enhances a clip and collects every enhanced frame (the clip's output must fit in memory; stream long clips instead).</summary>
    /// <param name="request">The frames to enhance.</param>
    /// <param name="options">Output size, frames per batch, seed, colour correction, ..., or <see langword="null"/> for the model's defaults.</param>
    /// <param name="cancellationToken">Stops the request between batches and stages.</param>
    /// <returns>The enhanced frames, one per input frame.</returns>
    public async Task<VideoEnhancementResult> EnhanceAsync(VideoEnhancementRequest request, VideoEnhancementOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var frames = new List<ImageInput>();
        VideoEnhancementUpdate? last = null;
        await foreach (var update in EnhanceStreamingAsync(request, options, cancellationToken).ConfigureAwait(false))
        {
            if (update.Frames is { } finished)
            {
                frames.AddRange(finished);
            }

            last = update;
        }

        if (last is not { Phase: VideoEnhancementPhase.Completed } || frames.Count == 0)
        {
            throw new InvalidOperationException("Video enhancement produced no result.");
        }

        return new VideoEnhancementResult(frames, last.FramesPerSecond, last.Seed);
    }

    /// <summary>
    /// Enhances a clip, reporting each stage of each batch of frames and handing out the finished frames batch by batch, in order; the
    /// last update (<see cref="VideoEnhancementPhase.Completed"/>) ends the clip. Frames of a <see cref="VideoEnhancementRequest.FrameSource"/>
    /// are read as the batches need them, and a consumer that falls behind holds the enhancer back after a few batches, so neither side
    /// collects the clip.
    /// </summary>
    /// <param name="request">The frames to enhance.</param>
    /// <param name="options">Output size, frames per batch, seed, colour correction, ..., or <see langword="null"/> for the model's defaults.</param>
    /// <param name="cancellationToken">Stops the request between batches and stages.</param>
    public IAsyncEnumerable<VideoEnhancementUpdate> EnhanceStreamingAsync(VideoEnhancementRequest request, VideoEnhancementOptions? options = null,
        CancellationToken cancellationToken = default);
}
