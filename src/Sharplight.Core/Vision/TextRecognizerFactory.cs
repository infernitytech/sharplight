using Sharplight.Models;
using Sharplight.Tasks;

namespace Sharplight.Vision;

/// <summary>Creates <see cref="TextRecognizerClient"/>s for <see cref="ModelKinds.Ocr"/> models with an image encoder, a decoder, a tokenizer and a feature extractor.</summary>
internal sealed class TextRecognizerFactory : ITaskClientFactory
{
    public string Kind => ModelKinds.Ocr;

    public Type ClientType => typeof(ITextRecognizer);

    public bool CanCreate(TaskModel model)
    {
        var description = model.Description;
        return description.TryGetComponent(ModelComponentRoles.Encoder, out var encoder) && encoder.Graphs.Contains(WellKnown.EncodeGraph)
            && description.TryGetComponent(ModelComponentRoles.Decoder, out var decoder) && decoder.Graphs.Contains(WellKnown.DecodeGraph) && decoder.Config.ContainsKey("startToken")
            && description.TryGetComponent(ModelComponentRoles.FeatureExtractor, out _)
            && description.TryGetComponent(ModelComponentRoles.Tokenizer, out _);
    }

    public ITaskClient Create(TaskModel model) => new TextRecognizerClient(model);
}
