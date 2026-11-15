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
}
