using Sharplight.Tasks;

namespace Sharplight.Vision;

/// <summary>
/// Monocular depth estimation (kind <see cref="Models.ModelKinds.DepthEstimation"/>): an image in, a depth map out (Depth Anything, DPT).
/// Follows the task-client pattern of every model kind (<see cref="ITaskClient"/>). Not thread-safe: run one request at a time.
/// </summary>
public interface IDepthEstimator : ITaskClient
{
    /// <summary>Estimates the depth of every pixel of an image.</summary>
    /// <param name="image">The image, any size.</param>
    /// <param name="options">Request settings, or <see langword="null"/> for the defaults.</param>
    /// <param name="cancellationToken">Cancels the request before it runs.</param>
    public Task<DepthMap> EstimateAsync(ImageInput image, DepthEstimationOptions? options = null, CancellationToken cancellationToken = default);
}
