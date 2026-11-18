using Sharplight.Models;
using Sharplight.Speech;
using Sharplight.Tasks;

namespace Sharplight.Architectures;

/// <summary>Creates the <see cref="ISpeechSynthesizer"/> of Qwen3-TTS models (built by <see cref="Qwen3TtsAdapter"/>).</summary>
internal sealed class Qwen3TtsSynthesizerFactory : ITaskClientFactory
{
    /// <inheritdoc/>
    public string Kind => ModelKinds.SpeechSynthesis;

    /// <inheritdoc/>
    public Type ClientType => typeof(ISpeechSynthesizer);

    /// <inheritdoc/>
    public bool CanCreate(TaskModel model) => Qwen3TtsSynthesizer.Handles(model);

    /// <inheritdoc/>
    public ITaskClient Create(TaskModel model) => new Qwen3TtsSynthesizer(model);
}
