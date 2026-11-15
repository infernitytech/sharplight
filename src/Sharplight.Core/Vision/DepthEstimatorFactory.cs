using Sharplight.Models;
using Sharplight.Tasks;

namespace Sharplight.Vision;

/// <summary>Creates <see cref="DepthEstimatorClient"/>s for <see cref="ModelKinds.DepthEstimation"/> models with a depth graph.</summary>
internal sealed class DepthEstimatorFactory : ITaskClientFactory
{
    public string Kind => ModelKinds.DepthEstimation;

    public Type ClientType => typeof(IDepthEstimator);

    public bool CanCreate(TaskModel model)
        => model.Description.Components.Any(c => c.Role == ModelComponentRoles.VisionEncoder && c.Graphs.Contains(WellKnown.DepthGraph))
            && model.Description.Components.Any(c => c.Role == ModelComponentRoles.FeatureExtractor);

    public ITaskClient Create(TaskModel model) => new DepthEstimatorClient(model);
}
