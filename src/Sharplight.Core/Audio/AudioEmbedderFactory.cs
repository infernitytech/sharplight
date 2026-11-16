using Sharplight.Models;
using Sharplight.Tasks;

namespace Sharplight.Audio;

/// <summary>Creates <see cref="IAudioEmbedder"/>s for <see cref="ModelKinds.AudioEmbedding"/> models with the <see cref="AudioEmbedderGraphs"/> graphs.</summary>
internal sealed class AudioEmbedderFactory : ITaskClientFactory
{
    public string Kind => ModelKinds.AudioEmbedding;

    public Type ClientType => typeof(IAudioEmbedder);

    public bool CanCreate(TaskModel model)
        => model.Description.Components.Any(c => c.Role == ModelComponentRoles.Encoder && c.Graphs.Contains(AudioEmbedderGraphs.AudioGraph))
            && model.Description.Components.Any(c => c.Role == ModelComponentRoles.FeatureExtractor);

    public ITaskClient Create(TaskModel model) => new AudioEmbedderClient(model);
}
