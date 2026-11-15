using Sharplight.Tasks;

namespace Sharplight.Vision;

/// <summary>
/// Image classification (kind <see cref="Models.ModelKinds.ImageClassification"/>): an image in, class scores out (ViT, DeiT, Swin, ConvNeXt
/// and other models with a classifier head). Follows the task-client pattern of every model kind (<see cref="ITaskClient"/>);
/// implementations come from <see cref="ITaskClientFactory"/> registrations. Not thread-safe: run one request at a time.
/// </summary>
public interface IImageClassifier : ITaskClient
{
    /// <summary>The class names, indexed by class.</summary>
    public IReadOnlyList<string> Labels { get; }

    /// <summary>Classifies one image.</summary>
    /// <param name="image">The image, any size; it is resized and normalized as the model's processor configuration says.</param>
    /// <param name="options">Request settings, or <see langword="null"/> for the defaults.</param>
    /// <param name="cancellationToken">Cancels the request before it runs.</param>
    public Task<ImageClassification> ClassifyAsync(ImageInput image, ImageClassificationOptions? options = null, CancellationToken cancellationToken = default);
}
