using Sharplight.Models;
using Sharplight.Speech;
using Sharplight.Tasks;

namespace Sharplight.Architectures;

/// <summary>Creates the <see cref="ISpeechSynthesizer"/> of Orpheus models (built by <see cref="OrpheusAdapter"/>).</summary>
internal sealed class OrpheusSynthesizerFactory : ITaskClientFactory
{
    /// <inheritdoc/>
    public string Kind => ModelKinds.SpeechSynthesis;

    /// <inheritdoc/>
    public Type ClientType => typeof(ISpeechSynthesizer);

    /// <inheritdoc/>
    public bool CanCreate(TaskModel model) => OrpheusSynthesizer.Handles(model);

    /// <inheritdoc/>
    public ITaskClient Create(TaskModel model) => new OrpheusSynthesizer(model);
}
