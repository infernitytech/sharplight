using Sharplight.Tasks;

namespace Sharplight.ImageGeneration;

/// <summary>
/// Image generation (kind <see cref="Models.ModelKinds.ImageGeneration"/>): a prompt and optionally an input image in, images out.
/// Follows the task-client pattern of every model kind (<see cref="ITaskClient"/>); corresponds to Microsoft.Extensions.AI's
/// <c>IImageGenerator</c>. Implementations (diffusion pipelines with their schedulers) come from architecture packages
/// (<see cref="ITaskClientFactory"/>). Not thread-safe: run one request at a time.
/// </summary>
public interface IImageGenerator : ITaskClient
{
    /// <summary>
    /// The model's own defaults (native size, steps, guidance, sampler and what it supports), which the <see langword="null"/> values of the
    /// per-request options resolve to.
    /// </summary>
    public ImageGenerationDefaults Defaults { get; }

    /// <summary>Generates images.</summary>
    /// <param name="request">The prompt and optional input image.</param>
    /// <param name="options">Size, steps, guidance, seed, ..., or <see langword="null"/> for the model's defaults.</param>
    /// <param name="cancellationToken">Stops generation between denoising steps.</param>
    public Task<ImageGenerationResult> GenerateAsync(ImageGenerationRequest request, ImageGenerationOptions? options = null,
        CancellationToken cancellationToken = default);

    /// <summary>Generates images, reporting each denoising step; the last update carries the finished images.</summary>
    /// <param name="request">The prompt and optional input image.</param>
    /// <param name="options">Size, steps, guidance, seed, ..., or <see langword="null"/> for the model's defaults.</param>
    /// <param name="cancellationToken">Stops generation between denoising steps.</param>
    public IAsyncEnumerable<ImageGenerationUpdate> GenerateStreamingAsync(ImageGenerationRequest request, ImageGenerationOptions? options = null,
        CancellationToken cancellationToken = default);
}
