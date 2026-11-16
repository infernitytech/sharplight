using Sharplight.Models;
using Sharplight.Tasks;

namespace Sharplight.Vision;

/// <summary>Creates <see cref="ImageSegmenterClient"/>s for <see cref="ModelKinds.ImageSegmentation"/> models with the segmentation graphs.</summary>
internal sealed class ImageSegmenterFactory : ITaskClientFactory
{
    public string Kind => ModelKinds.ImageSegmentation;

    public Type ClientType => typeof(IImageSegmenter);

    public bool CanCreate(TaskModel model)
        => model.Description.Components.Any(c => c.Role == ModelComponentRoles.VisionEncoder && c.Graphs.Contains(WellKnown.SegmentEncodeGraph))
            && model.Description.Components.Any(c => c.Role == ModelComponentRoles.MaskDecoder && c.Graphs.Contains(WellKnown.SegmentDecodeGraph))
            && model.Description.Components.Any(c => c.Role == ModelComponentRoles.FeatureExtractor);

    public ITaskClient Create(TaskModel model) => new ImageSegmenterClient(model);
}
