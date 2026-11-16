using Sharplight.Models;
using Sharplight.Tasks;

namespace Sharplight.Audio;

/// <summary>Creates <see cref="IMusicGenerator"/>s for <see cref="ModelKinds.MusicGeneration"/> models with the <see cref="MusicGenerationGraphs"/> graphs.</summary>
internal sealed class MusicGeneratorFactory : ITaskClientFactory
{
    public string Kind => ModelKinds.MusicGeneration;

    public Type ClientType => typeof(IMusicGenerator);

    public bool CanCreate(TaskModel model)
        => model.Description.Components.Any(c => c.Role == ModelComponentRoles.Decoder && c.Graphs.Contains(MusicGenerationGraphs.StepGraph))
            && model.Description.Components.Any(c => c.Role == ModelComponentRoles.TextEncoder && c.Graphs.Contains(MusicGenerationGraphs.TextGraph))
            && model.Description.Components.Any(c => c.Role == ModelComponentRoles.Vocoder && c.Graphs.Contains(AudioCodecGraphs.DecodeGraph));

    public ITaskClient Create(TaskModel model) => new MusicGeneratorClient(model);
}
