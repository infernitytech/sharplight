using Sharplight.Models;
using Sharplight.Tasks;

namespace Sharplight.Vision;

/// <summary>
/// Creates <see cref="ImageEncoderClient"/>s for models with an image-encoder graph: an <see cref="IImageClassifier"/> for
/// <see cref="ModelKinds.ImageClassification"/> models, and an <see cref="IImageEmbedder"/> for <see cref="ModelKinds.ImageEmbedding"/>
/// models and for classifiers (their trunk embeds too).
/// </summary>
/// <param name="kind">The model kind served.</param>
/// <param name="clientType">The client interface produced.</param>
internal sealed class ImageEncoderFactory(string kind, Type clientType) : ITaskClientFactory
{
    public string Kind => kind;

    public Type ClientType => clientType;

    public bool CanCreate(TaskModel model)
        => model.Description.Components.Any(c => c.Role == ModelComponentRoles.VisionEncoder && c.Graphs.Contains(WellKnown.ImageEncodeGraph))
            && model.Description.Components.Any(c => c.Role == ModelComponentRoles.FeatureExtractor);

    public ITaskClient Create(TaskModel model) => new ImageEncoderClient(model);
}
