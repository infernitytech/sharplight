using Sharplight.Tasks;

namespace Sharplight.Vision;

/// <summary>
/// Image super-resolution (kind <see cref="Models.ModelKinds.ImageUpscaling"/>): an image in, the same image at a larger size out (Real-ESRGAN). Large images are
/// processed in overlapping tiles that are blended seamlessly. Follows the task-client pattern of every model kind (<see cref="ITaskClient"/>). Requests are
/// serialized; the compute never runs on the calling thread.
/// </summary>
public interface IImageUpscaler : ITaskClient
{
    /// <summary>The factor by which the model enlarges every side.</summary>
    public int Scale { get; }

    /// <summary>Enlarges an image.</summary>
    /// <param name="image">The image, any size.</param>
    /// <param name="options">Request settings, or <see langword="null"/> for the defaults.</param>
    /// <param name="cancellationToken">Cancels the request, also between the tiles.</param>
    /// <returns>The image, <see cref="Scale"/> times as wide and high.</returns>
    public Task<ImageInput> UpscaleAsync(ImageInput image, ImageUpscalingOptions? options = null, CancellationToken cancellationToken = default);
}
