using Sharplight.Models;
using Sharplight.Speech;
using Sharplight.Tasks;

namespace Sharplight.Architectures;

/// <summary>Creates the <see cref="ISpeechSynthesizer"/> of CosyVoice 3 models (built by <see cref="CosyVoiceAdapter"/>).</summary>
internal sealed class CosyVoiceSynthesizerFactory : ITaskClientFactory
{
    /// <inheritdoc/>
    public string Kind => ModelKinds.SpeechSynthesis;

    /// <inheritdoc/>
    public Type ClientType => typeof(ISpeechSynthesizer);

    /// <inheritdoc/>
    public bool CanCreate(TaskModel model) => CosyVoiceSynthesizer.Handles(model);

    /// <inheritdoc/>
    public ITaskClient Create(TaskModel model) => new CosyVoiceSynthesizer(model);
}
