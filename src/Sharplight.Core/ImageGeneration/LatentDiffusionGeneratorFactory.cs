using Sharplight.Models;
using Sharplight.Tasks;

namespace Sharplight.ImageGeneration;

/// <summary>
/// The built-in image-generation implementation of the task layer: every <see cref="ModelKinds.ImageGeneration"/> model whose
/// description has text-encoder, denoiser, VAE, tokenizer and scheduler components with the <see cref="DiffusionGraphs"/> graphs gets a
/// <see cref="LatentDiffusionGenerator"/>. Model families with other pipelines register their own factory, which wins when it accepts
/// the model.
/// </summary>
internal sealed class LatentDiffusionGeneratorFactory : ITaskClientFactory
{
    public string Kind => ModelKinds.ImageGeneration;

    public Type ClientType => typeof(IImageGenerator);

    public bool CanCreate(TaskModel model)
    {
        var description = model.Description;
        return Has(description, ModelComponentRoles.TextEncoder, DiffusionGraphs.TextEncode)
               && Has(description, ModelComponentRoles.Denoiser, DiffusionGraphs.Denoise)
               && Has(description, ModelComponentRoles.Vae, DiffusionGraphs.VaeDecode)
               && description.TryGetComponent(ModelComponentRoles.Tokenizer, out _)
               && description.TryGetComponent(ModelComponentRoles.Scheduler, out _);
    }

    public ITaskClient Create(TaskModel model) => new LatentDiffusionGenerator(model);

    private static bool Has(ModelDescription description, string role, string graph)
        => description.TryGetComponent(role, out var component) && component.Graphs.Contains(graph);
}
