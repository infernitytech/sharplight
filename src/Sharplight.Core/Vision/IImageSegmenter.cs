using Sharplight.Tasks;

namespace Sharplight.Vision;

/// <summary>
/// Promptable image segmentation (kind <see cref="Models.ModelKinds.ImageSegmentation"/>): an image and a prompt (points and/or a box) in, masks of the
/// prompted object in pixels of that image out (SAM 2). The image stage is expensive and independent of the prompt: <see cref="EncodeImageAsync"/> does
/// it once, every following prompt is cheap (interactive use). Follows the task-client pattern of every model kind (<see cref="ITaskClient"/>).
/// Requests are serialized; the compute never runs on the calling thread.
/// </summary>
public interface IImageSegmenter : ITaskClient
{
    /// <summary>Runs the image stage of the model.</summary>
    /// <param name="image">The image, any size.</param>
    /// <param name="cancellationToken">Cancels the request before it runs.</param>
    public Task<SegmentationImage> EncodeImageAsync(ImageInput image, CancellationToken cancellationToken = default);

    /// <summary>Segments the object a prompt points at in an already encoded image.</summary>
    /// <param name="image">The encoded image (from this segmenter).</param>
    /// <param name="prompt">Points and/or a box, in pixels of the image.</param>
    /// <param name="options">Request settings, or <see langword="null"/> for the defaults.</param>
    /// <param name="cancellationToken">Cancels the request before it runs.</param>
    public Task<ImageSegmentation> SegmentAsync(SegmentationImage image, SegmentationPrompt prompt, SegmentationOptions? options = null, CancellationToken cancellationToken = default);

    /// <summary>Encodes an image and segments the object a prompt points at.</summary>
    /// <param name="image">The image, any size.</param>
    /// <param name="prompt">Points and/or a box, in pixels of the image.</param>
    /// <param name="options">Request settings, or <see langword="null"/> for the defaults.</param>
    /// <param name="cancellationToken">Cancels the request before it runs.</param>
    public async Task<ImageSegmentation> SegmentAsync(ImageInput image, SegmentationPrompt prompt, SegmentationOptions? options = null, CancellationToken cancellationToken = default)
        => await SegmentAsync(await EncodeImageAsync(image, cancellationToken).ConfigureAwait(false), prompt, options, cancellationToken).ConfigureAwait(false);
}
