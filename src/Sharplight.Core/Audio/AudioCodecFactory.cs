using Sharplight.Models;
using Sharplight.Tasks;

namespace Sharplight.Audio;

/// <summary>Creates <see cref="IAudioCodec"/>s for <see cref="ModelKinds.AudioCodec"/> models with the <see cref="AudioCodecGraphs"/> graphs.</summary>
internal sealed class AudioCodecFactory : ITaskClientFactory
{
    public string Kind => ModelKinds.AudioCodec;

    public Type ClientType => typeof(IAudioCodec);

    public bool CanCreate(TaskModel model)
        => model.Description.Components.Any(c => c.Role == ModelComponentRoles.Encoder && c.Graphs.Contains(AudioCodecGraphs.EncodeGraph))
            && model.Description.Components.Any(c => c.Role == ModelComponentRoles.Decoder && c.Graphs.Contains(AudioCodecGraphs.DecodeGraph));

    public ITaskClient Create(TaskModel model) => new AudioCodecClient(model);
}
