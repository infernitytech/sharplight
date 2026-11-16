using Sharplight.Models;
using Sharplight.Tasks;

namespace Sharplight.Audio;

/// <summary>Creates <see cref="IVoiceActivityDetector"/>s for <see cref="ModelKinds.VoiceActivity"/> models with the <see cref="VoiceActivityGraphs.StepGraph"/> graph.</summary>
internal sealed class VoiceActivityDetectorFactory : ITaskClientFactory
{
    public string Kind => ModelKinds.VoiceActivity;

    public Type ClientType => typeof(IVoiceActivityDetector);

    public bool CanCreate(TaskModel model)
        => model.Description.Components.Any(c => c.Role == ModelComponentRoles.Encoder && c.Graphs.Contains(VoiceActivityGraphs.StepGraph));

    public ITaskClient Create(TaskModel model) => new VoiceActivityDetectorClient(model);
}
