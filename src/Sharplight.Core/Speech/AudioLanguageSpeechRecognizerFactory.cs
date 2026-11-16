using Sharplight.Models;
using Sharplight.Tasks;

namespace Sharplight.Speech;

/// <summary>
/// The built-in speech-recognition implementation of the task layer for audio language models: every <see cref="ModelKinds.SpeechRecognition"/>
/// model whose encoder is an audio tower and whose decoder is a language model with an audio prompt gets an <see cref="AudioLanguageSpeechRecognizer"/>.
/// </summary>
internal sealed class AudioLanguageSpeechRecognizerFactory : ITaskClientFactory
{
    private readonly IServiceProvider? _services;

    /// <summary>Creates the factory.</summary>
    /// <param name="services">The container the recognizers resolve a voice activity detector from, or <see langword="null"/> outside dependency injection.</param>
    public AudioLanguageSpeechRecognizerFactory(IServiceProvider? services = null)
    {
        _services = services;
    }

    public string Kind => ModelKinds.SpeechRecognition;

    public Type ClientType => typeof(ISpeechRecognizer);

    public bool CanCreate(TaskModel model) => AudioLanguageSpeechRecognizer.Handles(model);

    public ITaskClient Create(TaskModel model) => new AudioLanguageSpeechRecognizer(model, _services);
}
