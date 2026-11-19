using Sharplight.Models;
using Sharplight.Tasks;

namespace Sharplight.Audio;

/// <summary>Creates <see cref="IAudioGenerator"/>s for <see cref="ModelKinds.AudioGeneration"/> models with the <see cref="AudioGenerationGraphs"/> graphs.</summary>
internal sealed class AudioGeneratorFactory : ITaskClientFactory
{
    public string Kind => ModelKinds.AudioGeneration;

    public Type ClientType => typeof(IAudioGenerator);

    public bool CanCreate(TaskModel model)
        => model.Description.Components.Any(c => c.Role == ModelComponentRoles.Denoiser && c.Graphs.Contains(AudioGenerationGraphs.DenoiseGraph))
            && model.Description.Components.Any(c => c.Role == ModelComponentRoles.TextEncoder && c.Graphs.Contains(AudioGenerationGraphs.TextGraph))
            && model.Description.Components.Any(c => c.Role == ModelComponentRoles.Encoder && c.Graphs.Contains(AudioGenerationGraphs.ConditionGraph))
            && model.Description.Components.Any(c => c.Role == ModelComponentRoles.Vae && c.Graphs.Contains(AudioGenerationGraphs.DecodeGraph));

    public ITaskClient Create(TaskModel model) => new AudioGeneratorClient(model);
}
