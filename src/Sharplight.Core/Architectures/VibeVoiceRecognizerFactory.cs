using Sharplight.Models;
using Sharplight.Speech;
using Sharplight.Tasks;

namespace Sharplight.Architectures;

/// <summary>Creates the <see cref="ISpeechRecognizer"/> of VibeVoice-ASR models (built by <see cref="VibeVoiceAsrAdapter"/>).</summary>
internal sealed class VibeVoiceRecognizerFactory : ITaskClientFactory
{
    /// <inheritdoc/>
    public string Kind => ModelKinds.SpeechRecognition;

    /// <inheritdoc/>
    public Type ClientType => typeof(ISpeechRecognizer);

    /// <inheritdoc/>
    public bool CanCreate(TaskModel model) => VibeVoiceRecognizer.Handles(model);

    /// <inheritdoc/>
    public ITaskClient Create(TaskModel model) => new VibeVoiceRecognizer(model);
}
