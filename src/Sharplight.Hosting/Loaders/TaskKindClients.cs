using Sharplight.ImageGeneration;
using Sharplight.Models;
using Sharplight.Speech;
using Sharplight.Tasks;
using Sharplight.VideoGeneration;
using Sharplight.Vision;

namespace Sharplight.Hosting.Loaders;

/// <summary>Which task client the runtime creates for the models of a kind. A new runnable kind is one more line here.</summary>
public static class TaskKindClients
{
    /// <summary>Whether models of <paramref name="kind"/> run through the task layer (the others are language models).</summary>
    /// <param name="kind">The model kind.</param>
    public static bool Handles(string kind) => kind is ModelKinds.SpeechRecognition or ModelKinds.SpeechSynthesis or ModelKinds.ImageGeneration
        or ModelKinds.VideoGeneration or ModelKinds.ImageClassification or ModelKinds.ImageEmbedding or ModelKinds.ObjectDetection
        or ModelKinds.DepthEstimation or ModelKinds.ImageSegmentation or ModelKinds.SemanticSegmentation or ModelKinds.Ocr or ModelKinds.ImageUpscaling;

    /// <summary>Creates the client of <paramref name="kind"/> for <paramref name="model"/>.</summary>
    /// <param name="provider">The registered task client factories.</param>
    /// <param name="model">The loaded model.</param>
    /// <param name="kind">The model kind.</param>
    /// <exception cref="NotSupportedException">No client is registered for the model.</exception>
    public static ITaskClient Create(TaskClientProvider provider, TaskModel model, string kind) => kind switch
    {
        ModelKinds.SpeechRecognition => provider.Create<ISpeechRecognizer>(model),
        ModelKinds.SpeechSynthesis => provider.Create<ISpeechSynthesizer>(model),
        ModelKinds.ImageGeneration => provider.Create<IImageGenerator>(model),
        ModelKinds.VideoGeneration => provider.Create<IVideoGenerator>(model),
        ModelKinds.ImageClassification => provider.Create<IImageClassifier>(model),
        ModelKinds.ImageEmbedding => provider.Create<IImageEmbedder>(model),
        ModelKinds.ObjectDetection => provider.Create<IObjectDetector>(model),
        ModelKinds.DepthEstimation => provider.Create<IDepthEstimator>(model),
        ModelKinds.ImageSegmentation => provider.Create<IImageSegmenter>(model),
        ModelKinds.SemanticSegmentation => provider.Create<ISemanticSegmenter>(model),
        ModelKinds.Ocr => provider.Create<ITextRecognizer>(model),
        ModelKinds.ImageUpscaling => provider.Create<IImageUpscaler>(model),
        _ => throw new NotSupportedException($"There is no task client for the kind '{kind}'."),
    };
}
