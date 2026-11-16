using Sharplight.Models;
using Sharplight.Tasks;

namespace Sharplight.Speech;

/// <summary>
/// The built-in speech-recognition implementation of the task layer for encoder-decoder models with a prompted decoder: every
/// <see cref="ModelKinds.SpeechRecognition"/> model with encoder, decoder, tokenizer and feature-extractor components whose decoder metadata
/// carries the control tokens (<see cref="SpeechKeys"/>) gets an <see cref="EncoderDecoderSpeechRecognizer"/>. Models with other decoders
/// (CTC, transducers) register their own factories.
/// </summary>
internal sealed class EncoderDecoderSpeechRecognizerFactory : ITaskClientFactory
{
    private readonly IServiceProvider? _services;

    /// <summary>Creates the factory.</summary>
    /// <param name="services">The container the recognizers resolve a voice activity detector from, or <see langword="null"/> outside dependency injection.</param>
    public EncoderDecoderSpeechRecognizerFactory(IServiceProvider? services = null)
    {
        _services = services;
    }

    public string Kind => ModelKinds.SpeechRecognition;

    public Type ClientType => typeof(ISpeechRecognizer);

    public bool CanCreate(TaskModel model)
    {
        var description = model.Description;
        return description.TryGetComponent(ModelComponentRoles.Encoder, out var encoder) && encoder.Graphs.Count > 0
            && description.TryGetComponent(ModelComponentRoles.Decoder, out var decoder) && decoder.Graphs.Count > 0
            && description.TryGetComponent(ModelComponentRoles.FeatureExtractor, out _)
            && description.TryGetComponent(ModelComponentRoles.Tokenizer, out _)
            && SpeechTokenProtocol.IsPresent(model.Metadata(ModelComponentRoles.Decoder));
    }

    public ITaskClient Create(TaskModel model) => new EncoderDecoderSpeechRecognizer(model, _services);
}
