using Sharplight.Tasks;

namespace Sharplight.VideoEnhancement;

/// <summary>
/// Video enhancement (kind <see cref="Models.ModelKinds.VideoEnhancement"/>): the frames of a clip (or one image) in, restored and usually
/// enlarged frames out (video super-resolution and restoration). Follows the task-client pattern of every model kind
/// (<see cref="ITaskClient"/>) and the shapes of <see cref="VideoGeneration.IVideoGenerator"/>. Implementations come from
/// <see cref="ITaskClientFactory"/> services of the architecture packages. Not thread-safe: run one request at a time. The work runs on a
/// background thread, never on the caller's thread.
/// </summary>
public interface IVideoEnhancer : ITaskClient
{
    /// <summary>
    /// The model's own defaults (native size, steps, guidance, sampler and what it supports), which the <see langword="null"/> values of the
    /// per-request options resolve to.
    /// </summary>
    public VideoEnhancementDefaults Defaults { get; }

    /// <summary>Enhances a clip.</summary>
    /// <param name="request">The frames to enhance.</param>
    /// <param name="options">Output size, frames per batch, seed, colour correction, ..., or <see langword="null"/> for the model's defaults.</param>
    /// <param name="cancellationToken">Stops the request between batches and stages.</param>
    /// <returns>The enhanced frames, one per input frame.</returns>
    public Task<VideoEnhancementResult> EnhanceAsync(VideoEnhancementRequest request, VideoEnhancementOptions? options = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Enhances a clip, reporting each stage of each batch of frames and delivering the finished frames batch by batch; the last update
    /// carries the whole result.
    /// </summary>
    /// <param name="request">The frames to enhance.</param>
    /// <param name="options">Output size, frames per batch, seed, colour correction, ..., or <see langword="null"/> for the model's defaults.</param>
    /// <param name="cancellationToken">Stops the request between batches and stages.</param>
    public IAsyncEnumerable<VideoEnhancementUpdate> EnhanceStreamingAsync(VideoEnhancementRequest request, VideoEnhancementOptions? options = null,
        CancellationToken cancellationToken = default);
}
