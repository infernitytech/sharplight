using Sharplight.Models;
using Sharplight.Tasks;

namespace Sharplight.Audio;

/// <summary>Creates <see cref="IAudioSeparator"/>s for <see cref="ModelKinds.AudioSeparation"/> models with the <see cref="SeparationGraphs"/> graph.</summary>
internal sealed class AudioSeparatorFactory : ITaskClientFactory
{
    public string Kind => ModelKinds.AudioSeparation;

    public Type ClientType => typeof(IAudioSeparator);

    public bool CanCreate(TaskModel model)
        => model.Description.Components.Any(c => c.Role == ModelComponentRoles.Encoder && c.Graphs.Contains(SeparationGraphs.SeparateGraph));

    public ITaskClient Create(TaskModel model) => new AudioSeparatorClient(model);
}
