using Sharplight.Tasks;

namespace Sharplight.VideoGeneration;

/// <summary>
/// Video generation (kind <see cref="Models.ModelKinds.VideoGeneration"/>): a prompt and optionally a start image in, video frames out
/// (text-to-video and image-to-video). Follows the task-client pattern of every model kind (<see cref="ITaskClient"/>) and mirrors
/// <see cref="ImageGeneration.IImageGenerator"/>. Implementations (video diffusion pipelines) come from <see cref="ITaskClientFactory"/>
/// services. Not thread-safe: run one request at a time. The work runs on the thread pool, never on the caller's thread.
/// </summary>
public interface IVideoGenerator : ITaskClient
{
    /// <summary>
    /// The model's own defaults (native size, steps, guidance, sampler and what it supports), which the <see langword="null"/> values of the
    /// per-request options resolve to.
    /// </summary>
    public VideoGenerationDefaults Defaults { get; }

    /// <summary>Generates a video.</summary>
    /// <param name="request">The prompt and optional start image.</param>
    /// <param name="options">Size, frames, steps, guidance, seed, ..., or <see langword="null"/> for the model's defaults.</param>
    /// <param name="cancellationToken">Stops generation between denoising steps and decoded frames.</param>
    public Task<VideoGenerationResult> GenerateAsync(VideoGenerationRequest request, VideoGenerationOptions? options = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Generates a video, reporting each denoising step (with preview frames when asked) and each decoded chunk of frames; the last
    /// update carries the finished video.
    /// </summary>
    /// <param name="request">The prompt and optional start image.</param>
    /// <param name="options">Size, frames, steps, guidance, seed, ..., or <see langword="null"/> for the model's defaults.</param>
    /// <param name="cancellationToken">Stops generation between denoising steps and decoded frames.</param>
    public IAsyncEnumerable<VideoGenerationUpdate> GenerateStreamingAsync(VideoGenerationRequest request, VideoGenerationOptions? options = null,
        CancellationToken cancellationToken = default);
}
