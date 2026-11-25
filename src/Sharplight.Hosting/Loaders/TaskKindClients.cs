using System.Collections.Frozen;
using Sharplight.Audio;
using Sharplight.ImageGeneration;
using Sharplight.Llm.Embeddings;
using Sharplight.Llm.Serving.Hosting;
using Sharplight.Models;
using Sharplight.Speech;
using Sharplight.Tasks;
using Sharplight.VideoEnhancement;
using Sharplight.VideoGeneration;
using Sharplight.Vision;

namespace Sharplight.Hosting.Loaders;

/// <summary>Which task client the runtime creates for the models of a kind. A new runnable kind is one more entry in <see cref="ClientTypes"/>.</summary>
public static class TaskKindClients
{
    /// <summary>The client interface created for each kind that runs through the task layer (the other kinds are language models).</summary>
    public static IReadOnlyDictionary<string, Type> ClientTypes { get; } = new Dictionary<string, Type>(StringComparer.Ordinal)
    {
        [ModelKinds.SpeechRecognition] = typeof(ISpeechRecognizer),
        [ModelKinds.SpeechSynthesis] = typeof(ISpeechSynthesizer),
        [ModelKinds.ImageGeneration] = typeof(IImageGenerator),
        [ModelKinds.VideoGeneration] = typeof(IVideoGenerator),
        [ModelKinds.VideoEnhancement] = typeof(IVideoEnhancer),
        [ModelKinds.ImageClassification] = typeof(IImageClassifier),
        [ModelKinds.ImageEmbedding] = typeof(IImageEmbedder),
        [ModelKinds.ObjectDetection] = typeof(IObjectDetector),
        [ModelKinds.DepthEstimation] = typeof(IDepthEstimator),
        [ModelKinds.ImageSegmentation] = typeof(IImageSegmenter),
        [ModelKinds.SemanticSegmentation] = typeof(ISemanticSegmenter),
        [ModelKinds.Ocr] = typeof(ITextRecognizer),
        [ModelKinds.ImageUpscaling] = typeof(IImageUpscaler),
        [ModelKinds.AudioCodec] = typeof(IAudioCodec),
        [ModelKinds.VoiceActivity] = typeof(IVoiceActivityDetector),
        [ModelKinds.AudioEmbedding] = typeof(IAudioEmbedder),
        [ModelKinds.MusicGeneration] = typeof(IMusicGenerator),
        [ModelKinds.AudioGeneration] = typeof(IAudioGenerator),
        [ModelKinds.AudioSeparation] = typeof(IAudioSeparator),
        [ModelKinds.TextEmbedding] = typeof(ITextEmbedder),
        [ModelKinds.TextReranking] = typeof(ITextReranker),
    }.ToFrozenDictionary(StringComparer.Ordinal);

    /// <summary>
    /// The host kind a model of task kind <paramref name="kind"/> is registered as: <see cref="ModelKind.Embedding"/> for text embedding
    /// and <see cref="ModelKind.Reranker"/> for reranking models (the kinds the retrieval endpoints serve), else <see cref="ModelKind.Task"/>.
    /// </summary>
    /// <param name="kind">The task kind (<see cref="ModelKinds"/>).</param>
    public static ModelKind HostedKind(string kind) => kind switch
    {
        ModelKinds.TextEmbedding => ModelKind.Embedding,
        ModelKinds.TextReranking => ModelKind.Reranker,
        _ => ModelKind.Task,
    };

    /// <summary>Whether models of <paramref name="kind"/> run through the task layer (the others are language models).</summary>
    /// <param name="kind">The model kind.</param>
    public static bool Handles(string kind) => ClientTypes.ContainsKey(kind);

    /// <summary>Creates the client of <paramref name="kind"/> for <paramref name="model"/>.</summary>
    /// <param name="provider">The registered task client factories.</param>
    /// <param name="model">The loaded model.</param>
    /// <param name="kind">The model kind.</param>
    /// <exception cref="NotSupportedException">No client is registered for the model.</exception>
    public static ITaskClient Create(TaskClientProvider provider, TaskModel model, string kind)
    {
        ArgumentNullException.ThrowIfNull(provider);
        return ClientTypes.TryGetValue(kind, out var clientType)
            ? provider.Create(clientType, model)
            : throw new NotSupportedException($"There is no task client for the kind '{kind}'.");
    }
}
