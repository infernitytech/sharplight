namespace Sharplight.Models;

/// <summary>
/// Well-known model kinds (<see cref="ModelDescription.Kind"/>). The set is open-ended: a kind is any lowercase, hyphenated string,
/// so new model classes (object detection, depth, audio codecs, time series, ...) use their own kind without changing the package
/// format, this list or any other Core type. The kinds listed here are the ones the Core task layer has client abstractions for.
/// </summary>
public static class ModelKinds
{
    /// <summary>Autoregressive or masked-diffusion text generation (<c>Sharplight.Llm.ITextGenerator</c>, <c>LlmModel</c>).</summary>
    public const string TextGeneration = "text-generation";

    /// <summary>Audio in, text with timestamps out (<c>Sharplight.Speech.ISpeechRecognizer</c>).</summary>
    public const string SpeechRecognition = "speech-recognition";

    /// <summary>Text in, audio out (<c>Sharplight.Speech.ISpeechSynthesizer</c>).</summary>
    public const string SpeechSynthesis = "speech-synthesis";

    /// <summary>Text and optionally an image in, images out (<c>Sharplight.ImageGeneration.IImageGenerator</c>).</summary>
    public const string ImageGeneration = "image-generation";

    /// <summary>Text and optionally a start image in, video frames out (<c>Sharplight.VideoGeneration.IVideoGenerator</c>).</summary>
    public const string VideoGeneration = "video-generation";

    /// <summary>Image in, class scores out (<c>Sharplight.Vision.IImageClassifier</c>).</summary>
    public const string ImageClassification = "image-classification";

    /// <summary>Image (and optionally text) in, embedding vectors out; zero-shot classification with a text tower (<c>Sharplight.Vision.IImageEmbedder</c>).</summary>
    public const string ImageEmbedding = "image-embedding";

    /// <summary>Image in, labelled boxes out.</summary>
    public const string ObjectDetection = "object-detection";

    /// <summary>Image (and optionally prompts) in, masks out: semantic, instance or promptable segmentation.</summary>
    public const string ImageSegmentation = "image-segmentation";

    /// <summary>Image in, per-pixel depth out.</summary>
    public const string DepthEstimation = "depth-estimation";

    /// <summary>Image in, recognized text out.</summary>
    public const string Ocr = "ocr";

    /// <summary>Image in, larger image out.</summary>
    public const string ImageUpscaling = "image-upscaling";
}
