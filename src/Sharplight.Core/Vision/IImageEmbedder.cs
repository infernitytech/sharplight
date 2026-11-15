using Sharplight.Tasks;

namespace Sharplight.Vision;

/// <summary>
/// Image embeddings (kind <see cref="Models.ModelKinds.ImageEmbedding"/>): an image in, a feature vector out (DINOv2, CLIP and SigLIP image
/// towers, ViT trunks). Models with a text tower (CLIP, SigLIP) also embed text into the same space and classify zero-shot. Follows the
/// task-client pattern of every model kind (<see cref="ITaskClient"/>). Not thread-safe: run one request at a time.
/// </summary>
public interface IImageEmbedder : ITaskClient
{
    /// <summary>The length of an embedding vector.</summary>
    public int Dimensions { get; }

    /// <summary>True when the model has a text tower (<see cref="EmbedTextAsync"/> and <see cref="ClassifyZeroShotAsync"/> work).</summary>
    public bool SupportsText { get; }

    /// <summary>Embeds one image.</summary>
    /// <param name="image">The image, any size.</param>
    /// <param name="options">Request settings, or <see langword="null"/> for the defaults.</param>
    /// <param name="cancellationToken">Cancels the request before it runs.</param>
    public Task<float[]> EmbedImageAsync(ImageInput image, ImageEmbeddingOptions? options = null, CancellationToken cancellationToken = default);

    /// <summary>Embeds a text into the image embedding space.</summary>
    /// <param name="text">The text (a caption or a class prompt such as "a photo of a dog").</param>
    /// <param name="options">Request settings, or <see langword="null"/> for the defaults.</param>
    /// <param name="cancellationToken">Cancels the request before it runs.</param>
    /// <exception cref="NotSupportedException">The model has no text tower.</exception>
    public Task<float[]> EmbedTextAsync(string text, ImageEmbeddingOptions? options = null, CancellationToken cancellationToken = default);

    /// <summary>Scores an image against candidate class prompts (zero-shot classification).</summary>
    /// <param name="image">The image.</param>
    /// <param name="candidates">The class prompts, used as labels of the result.</param>
    /// <param name="options">Request settings, or <see langword="null"/> for the defaults.</param>
    /// <param name="cancellationToken">Cancels the request before it runs.</param>
    /// <exception cref="NotSupportedException">The model has no text tower.</exception>
    public Task<ImageClassification> ClassifyZeroShotAsync(ImageInput image, IReadOnlyList<string> candidates, ImageClassificationOptions? options = null,
        CancellationToken cancellationToken = default);
}
