using Sharplight.Models;
using Sharplight.Speech;
using Sharplight.Tasks;

namespace Sharplight.Architectures;

/// <summary>Creates the <see cref="ISpeechSynthesizer"/> of VibeVoice models (built by <see cref="VibeVoiceAdapter"/>).</summary>
internal sealed class VibeVoiceSynthesizerFactory : ITaskClientFactory
{
    /// <inheritdoc/>
    public string Kind => ModelKinds.SpeechSynthesis;

    /// <inheritdoc/>
    public Type ClientType => typeof(ISpeechSynthesizer);

    /// <inheritdoc/>
    public bool CanCreate(TaskModel model) => VibeVoiceSynthesizer.Handles(model);

    /// <inheritdoc/>
    public ITaskClient Create(TaskModel model) => new VibeVoiceSynthesizer(model);
}
