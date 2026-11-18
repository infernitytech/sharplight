using Sharplight.Models;
using Sharplight.Tasks;

namespace Sharplight.Speech;

/// <summary>
/// The built-in speech-recognition implementation of the task layer for causal streaming audio language models (Voxtral Realtime): every
/// <see cref="ModelKinds.SpeechRecognition"/> model whose encoder is a causal audio tower gets a <see cref="CausalAudioSpeechRecognizer"/>.
/// </summary>
internal sealed class CausalAudioSpeechRecognizerFactory : ITaskClientFactory
{
    private readonly IServiceProvider? _services;

    /// <summary>Creates the factory.</summary>
    /// <param name="services">The container the recognizers resolve a voice activity detector from, or <see langword="null"/> outside dependency injection.</param>
    public CausalAudioSpeechRecognizerFactory(IServiceProvider? services = null)
    {
        _services = services;
    }

    public string Kind => ModelKinds.SpeechRecognition;

    public Type ClientType => typeof(ISpeechRecognizer);

    public bool CanCreate(TaskModel model) => CausalAudioSpeechRecognizer.Handles(model);

    public ITaskClient Create(TaskModel model) => new CausalAudioSpeechRecognizer(model, _services);
}
