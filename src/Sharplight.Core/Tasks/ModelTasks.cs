namespace Sharplight.Tasks;

/// <summary>
/// Well-known task ids (<see cref="ModelTaskDescriptor.Kind"/>). The set is open-ended: a task id is any lowercase, hyphenated
/// string, so new model classes (object detection, depth, audio codecs, time series, ...) add their own id without changing the
/// container format or this list. The ids here are the ones the core task layer has abstractions for.
/// </summary>
public static class ModelTasks
{
    /// <summary>Autoregressive or diffusion text generation (<c>Sharplight.Llm.LlmModel</c>).</summary>
    public const string TextGeneration = "text-generation";

    /// <summary>Sentence embeddings (<c>Sharplight.Llm.Embeddings.EmbeddingModel</c>).</summary>
    public const string Embedding = "embedding";

    /// <summary>Query / document relevance scoring (<c>Sharplight.Llm.Embeddings.Reranker</c>).</summary>
    public const string Reranking = "reranking";

    /// <summary>Audio in, text with timestamps out (<see cref="Speech.ISpeechRecognizer"/>).</summary>
    public const string SpeechRecognition = "speech-recognition";

    /// <summary>Text in, audio out (<see cref="Speech.ISpeechSynthesizer"/>).</summary>
    public const string SpeechSynthesis = "speech-synthesis";

    /// <summary>Text and optionally an image in, images out (<see cref="ImageGeneration.IImageGenerator"/>).</summary>
    public const string ImageGeneration = "image-generation";
}
