using Sharplight.Models;
using Sharplight.Tasks;
using Sharplight.VideoEnhancement;

namespace Sharplight.Architectures.SeedVr;

/// <summary>Creates a <see cref="SeedVrEnhancer"/> for every <see cref="ModelKinds.VideoEnhancement"/> model built by <see cref="SeedVrAdapter"/>.</summary>
internal sealed class SeedVrEnhancerFactory : ITaskClientFactory
{
    public string Kind => ModelKinds.VideoEnhancement;

    public Type ClientType => typeof(IVideoEnhancer);

    public bool CanCreate(TaskModel model)
        => model.Definition.Architecture == SeedVrAdapter.Architecture
           && model.Description.TryGetComponent(ModelComponentRoles.Transformer, out var transformer) && transformer.Graphs.Contains(SeedVrGraphs.Restore)
           && model.Description.TryGetComponent(ModelComponentRoles.Vae, out var vae) && vae.Graphs.Contains(SeedVrGraphs.VaeDecode);

    public ITaskClient Create(TaskModel model) => new SeedVrEnhancer(model);
}
