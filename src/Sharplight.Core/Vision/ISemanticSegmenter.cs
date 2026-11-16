using Sharplight.Tasks;

namespace Sharplight.Vision;

/// <summary>
/// Per-pixel segmentation (kind <see cref="Models.ModelKinds.SemanticSegmentation"/>): an image in, a class label for every pixel out (SegFormer, and matting
/// models that return an alpha matte). Follows the task-client pattern of every model kind (<see cref="ITaskClient"/>). Not thread-safe: run one request at a time.
/// </summary>
public interface ISemanticSegmenter : ITaskClient
{
    /// <summary>The class names, by class index (empty when the model has none).</summary>
    public IReadOnlyList<string> ClassNames { get; }

    /// <summary>Labels every pixel of an image.</summary>
    /// <param name="image">The image, any size.</param>
    /// <param name="options">Request settings, or <see langword="null"/> for the defaults.</param>
    /// <param name="cancellationToken">Cancels the request before it runs.</param>
    public Task<SemanticSegmentation> SegmentAsync(ImageInput image, SemanticSegmentationOptions? options = null, CancellationToken cancellationToken = default);
}
