using Sharplight.Models;
using Sharplight.Tasks;

namespace Sharplight.Audio;

/// <summary>Creates <see cref="IAudioGenerator"/>s for <see cref="ModelKinds.AudioGeneration"/> models with the rectified-flow form of the <see cref="TimedAudioDiffusionGraphs"/> graphs.</summary>
internal sealed class FlowAudioDiffusionFactory : ITaskClientFactory
{
    public string Kind => ModelKinds.AudioGeneration;

    public Type ClientType => typeof(IAudioGenerator);

    public bool CanCreate(TaskModel model)
        => model.Description.Components.Any(c => c.Role == ModelComponentRoles.Denoiser && c.Graphs.Contains(TimedAudioDiffusionGraphs.DenoiseGraph)
                && c.ConfigValue(TimedAudioDiffusionGraphs.ObjectiveKey, "v") is "rectified_flow" or "rf_denoiser")
            && model.Description.Components.Any(c => c.Role == ModelComponentRoles.TextEncoder && c.Graphs.Contains(TimedAudioDiffusionGraphs.TextGraph))
            && model.Description.Components.Any(c => c.Role == ModelComponentRoles.Encoder && c.Graphs.Contains(TimedAudioDiffusionGraphs.TimingGraph))
            && model.Description.Components.Any(c => c.Role == ModelComponentRoles.Vae && c.Graphs.Contains(AudioGenerationGraphs.DecodeGraph));

    public ITaskClient Create(TaskModel model) => new FlowAudioDiffusionClient(model);
}
