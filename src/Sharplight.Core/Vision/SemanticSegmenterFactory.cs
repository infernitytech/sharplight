using Sharplight.Models;
using Sharplight.Tasks;

namespace Sharplight.Vision;

/// <summary>Creates <see cref="SemanticSegmenterClient"/>s for <see cref="ModelKinds.SemanticSegmentation"/> models with the semantic segmentation graph.</summary>
internal sealed class SemanticSegmenterFactory : ITaskClientFactory
{
    public string Kind => ModelKinds.SemanticSegmentation;

    public Type ClientType => typeof(ISemanticSegmenter);

    public bool CanCreate(TaskModel model)
        => model.Description.Components.Any(c => c.Role == ModelComponentRoles.VisionEncoder && c.Graphs.Contains(WellKnown.SemanticSegmentGraph))
            && model.Description.Components.Any(c => c.Role == ModelComponentRoles.FeatureExtractor);

    public ITaskClient Create(TaskModel model) => new SemanticSegmenterClient(model);
}
