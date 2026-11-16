using Sharplight.Models;
using Sharplight.Tasks;

namespace Sharplight.Speech;

/// <summary>
/// The built-in speech-recognition implementation of the task layer for models with a frame-aligned decoder: every
/// <see cref="ModelKinds.SpeechRecognition"/> model with encoder, tokenizer and feature-extractor components whose decoder component configures
/// CTC, RNN-T or TDT decoding (<see cref="FrameDecodingSettings"/>) gets a <see cref="FrameSpeechRecognizer"/>.
/// </summary>
internal sealed class FrameSpeechRecognizerFactory : ITaskClientFactory
{
    public string Kind => ModelKinds.SpeechRecognition;

    public Type ClientType => typeof(ISpeechRecognizer);

    public bool CanCreate(TaskModel model)
    {
        var description = model.Description;
        return description.TryGetComponent(ModelComponentRoles.Encoder, out var encoder) && encoder.Graphs.Count > 0
            && description.TryGetComponent(ModelComponentRoles.Decoder, out var decoder) && FrameDecodingSettings.IsPresent(decoder)
            && description.TryGetComponent(ModelComponentRoles.FeatureExtractor, out _)
            && description.TryGetComponent(ModelComponentRoles.Tokenizer, out _);
    }

    public ITaskClient Create(TaskModel model) => new FrameSpeechRecognizer(model);
}
