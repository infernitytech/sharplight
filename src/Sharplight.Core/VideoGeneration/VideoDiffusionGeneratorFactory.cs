using Sharplight.Models;
using Sharplight.Tasks;

namespace Sharplight.VideoGeneration;

/// <summary>
/// The built-in video-generation implementation of the task layer: every <see cref="ModelKinds.VideoGeneration"/> model whose
/// description has text-encoder, transformer, VAE, tokenizer and scheduler components with the <see cref="VideoDiffusionGraphs"/>
/// graphs gets a <see cref="VideoDiffusionGenerator"/>.
/// </summary>
internal sealed class VideoDiffusionGeneratorFactory : ITaskClientFactory
{
    public string Kind => ModelKinds.VideoGeneration;

    public Type ClientType => typeof(IVideoGenerator);

    public bool CanCreate(TaskModel model)
    {
        var description = model.Description;
        return Has(description, ModelComponentRoles.TextEncoder, VideoDiffusionGraphs.TextEncode)
               && Has(description, ModelComponentRoles.Transformer, VideoDiffusionGraphs.Denoise)
               && Has(description, ModelComponentRoles.Vae, VideoDiffusionGraphs.VaeDecodeFirst)
               && Has(description, ModelComponentRoles.Vae, VideoDiffusionGraphs.VaeDecodeNext)
               && description.TryGetComponent(ModelComponentRoles.Tokenizer, out _)
               && description.TryGetComponent(ModelComponentRoles.Scheduler, out _);
    }

    public ITaskClient Create(TaskModel model) => new VideoDiffusionGenerator(model);

    private static bool Has(ModelDescription description, string role, string graph)
        => description.TryGetComponent(role, out var component) && component.Graphs.Contains(graph);
}
