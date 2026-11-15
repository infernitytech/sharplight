using Sharplight.Tasks;

namespace Sharplight.Vision;

/// <summary>
/// Object detection (kind <see cref="Models.ModelKinds.ObjectDetection"/>): an image in, labelled boxes in pixels of that image out (DETR, RT-DETR).
/// Follows the task-client pattern of every model kind (<see cref="ITaskClient"/>). Not thread-safe: run one request at a time.
/// </summary>
public interface IObjectDetector : ITaskClient
{
    /// <summary>The class names, by class index.</summary>
    public IReadOnlyList<string> Labels { get; }

    /// <summary>Finds the objects in an image.</summary>
    /// <param name="image">The image, any size.</param>
    /// <param name="options">Request settings, or <see langword="null"/> for the defaults.</param>
    /// <param name="cancellationToken">Cancels the request before it runs.</param>
    public Task<ObjectDetection> DetectAsync(ImageInput image, ObjectDetectionOptions? options = null, CancellationToken cancellationToken = default);
}
