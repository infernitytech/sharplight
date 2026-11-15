using Sharplight.Hub;

namespace Sharplight.Studio.Models;

/// <summary>The names Studio shows for <see cref="ModelTask"/> groups.</summary>
internal static class ModelTaskTitles
{
    /// <summary>The name of a task in the filter and on result cards.</summary>
    /// <param name="task">The task.</param>
    public static string Title(ModelTask task) => task switch
    {
        ModelTask.TextGeneration => "Text generation",
        ModelTask.VisionLanguage => "Vision-language",
        ModelTask.SpeechToText => "Speech to text",
        ModelTask.TextToSpeech => "Text to speech",
        ModelTask.ImageGeneration => "Image generation",
        ModelTask.VideoGeneration => "Video generation",
        ModelTask.ImageClassification => "Image classification",
        ModelTask.ObjectDetection => "Object detection",
        ModelTask.Segmentation => "Segmentation",
        ModelTask.DepthEstimation => "Depth estimation",
        ModelTask.Ocr => "OCR",
        ModelTask.Upscaling => "Upscaling",
        ModelTask.Audio => "Audio",
        ModelTask.Embeddings => "Embeddings and retrieval",
        ModelTask.TimeSeries => "Time series",
        ModelTask.ThreeD => "3D",
        ModelTask.Bio => "Bio",
        ModelTask.Robotics => "Robotics",
        _ => "Other",
    };
}
