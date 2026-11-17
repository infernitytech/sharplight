using Sharplight.Models;
using Sharplight.Tasks;

namespace Sharplight.Vision;

/// <summary>Creates <see cref="ObjectDetectorClient"/>s for <see cref="ModelKinds.ObjectDetection"/> models with a detection graph.</summary>
internal sealed class ObjectDetectorFactory : ITaskClientFactory
{
    public string Kind => ModelKinds.ObjectDetection;

    public Type ClientType => typeof(IObjectDetector);

    public bool CanCreate(TaskModel model)
        => model.Description.Components.Any(c => c.Role == ModelComponentRoles.VisionEncoder && c.Graphs.Contains(WellKnown.DetectGraph))
            && model.Description.Components.Any(c => c.Role == ModelComponentRoles.FeatureExtractor);

    public ITaskClient Create(TaskModel model)
        => model.Description.Component(ModelComponentRoles.VisionEncoder).Config.ContainsKey(DenseDetectorClient.HeadKey) ? new DenseDetectorClient(model) : new ObjectDetectorClient(model);
}
