using Sharplight.Models;
using Sharplight.Speech;
using Sharplight.Tasks;

namespace Sharplight.Architectures;

/// <summary>Creates the <see cref="ISpeechSynthesizer"/> of Chatterbox models (built by <see cref="ChatterboxAdapter"/>).</summary>
internal sealed class ChatterboxSynthesizerFactory : ITaskClientFactory
{
    /// <inheritdoc/>
    public string Kind => ModelKinds.SpeechSynthesis;

    /// <inheritdoc/>
    public Type ClientType => typeof(ISpeechSynthesizer);

    /// <inheritdoc/>
    public bool CanCreate(TaskModel model) => ChatterboxSynthesizer.Handles(model);

    /// <inheritdoc/>
    public ITaskClient Create(TaskModel model) => new ChatterboxSynthesizer(model);
}
