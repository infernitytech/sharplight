using Sharplight.Models;
using Sharplight.Tasks;

namespace Sharplight.Vision;

/// <summary>Creates <see cref="ImageUpscalerClient"/>s for <see cref="ModelKinds.ImageUpscaling"/> models with an upscale graph.</summary>
internal sealed class ImageUpscalerFactory : ITaskClientFactory
{
    public string Kind => ModelKinds.ImageUpscaling;

    public Type ClientType => typeof(IImageUpscaler);

    public bool CanCreate(TaskModel model)
        => model.Description.Components.Any(c => c.Role == ModelComponentRoles.VisionEncoder && c.Graphs.Contains(WellKnown.UpscaleGraph));

    public ITaskClient Create(TaskModel model) => new ImageUpscalerClient(model);
}
