using Sharplight.Models;
using Sharplight.Tasks;

namespace Sharplight.Vision;

/// <summary>
/// Creates <see cref="LanguageModelTextRecognizerClient"/>s for decoder-only vision-language models that read text from images: a text-generation model with a
/// vision tower and recognition tasks (<see cref="CanonicalKeys.RecognitionTasks"/>) in its metadata. The model keeps its kind (it is still a chat model); the factory
/// adds the <see cref="ITextRecognizer"/> view of it.
/// </summary>
internal sealed class LanguageModelTextRecognizerFactory : ITaskClientFactory
{
    public string Kind => ModelKinds.TextGeneration;

    public Type ClientType => typeof(ITextRecognizer);

    public bool CanCreate(TaskModel model)
        => model.Definition.Graphs.ContainsKey(WellKnown.DecodeMultimodalGraph)
           && model.Definition.Graphs.ContainsKey(WellKnown.VisionEncodeGraph)
           && model.Definition.SourceMetadata.ContainsKey(CanonicalKeys.RecognitionTasks);

    public ITaskClient Create(TaskModel model) => new LanguageModelTextRecognizerClient(model);
}
