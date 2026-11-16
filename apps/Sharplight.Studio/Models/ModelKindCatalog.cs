using Sharplight.Hub;
using Sharplight.Models;

namespace Sharplight.Studio.Models;

/// <summary>
/// The model kinds Studio knows, with their presentation. A kind is the open-ended string of a model description, so a kind that is
/// not listed gets a generic entry (<see cref="For(string?)"/>). Which kinds are <see cref="ModelKindInfo.Runnable"/> is the one place
/// that changes when the engine learns a new kind.
/// </summary>
internal static class ModelKindCatalog
{
    /// <summary>The kind of embedding models (a package describes them as text generation; its graphs tell them apart).</summary>
    public const string Embeddings = "embeddings";

    /// <summary>The kind of models that fit no other.</summary>
    public const string Other = "other";

    /// <summary>Every kind Studio knows, in the order the filters list them.</summary>
    public static IReadOnlyList<ModelKindInfo> All { get; } =
    [
        new(ModelKinds.TextGeneration, "Text generation", "IconChat", [ModelTask.TextGeneration, ModelTask.VisionLanguage], Runnable: true),
        new(ModelKinds.SpeechRecognition, "Speech to text", "IconMic", [ModelTask.SpeechToText], Runnable: true),
        new(ModelKinds.SpeechSynthesis, "Text to speech", "IconSpeaker", [ModelTask.TextToSpeech], Runnable: true),
        new(ModelKinds.ImageGeneration, "Image generation", "IconImage", [ModelTask.ImageGeneration], Runnable: true),
        new("video-generation", "Video generation", "IconVideo", [ModelTask.VideoGeneration], Runnable: false),
        new(ModelKinds.ImageClassification, "Image classification", "IconTag", [ModelTask.ImageClassification], Runnable: true),
        new(ModelKinds.ImageEmbedding, "Image embedding", "IconEmbed", [], Runnable: true),
        new(ModelKinds.ObjectDetection, "Object detection", "IconScan", [ModelTask.ObjectDetection], Runnable: true),
        new(ModelKinds.ImageSegmentation, "Segmentation", "IconSegment", [ModelTask.Segmentation], Runnable: true),
        new(ModelKinds.DepthEstimation, "Depth estimation", "IconLayers", [ModelTask.DepthEstimation], Runnable: true),
        new(ModelKinds.Ocr, "OCR", "IconDocument", [ModelTask.Ocr], Runnable: true),
        new(ModelKinds.ImageUpscaling, "Upscaling", "IconUpscale", [ModelTask.Upscaling], Runnable: true),
        new("audio", "Audio", "IconWave", [ModelTask.Audio], Runnable: false),
        new(ModelKinds.AudioCodec, "Audio codec", "IconWave", [], Runnable: true),
        new(ModelKinds.VoiceActivity, "Voice activity", "IconWave", [], Runnable: true),
        new(Embeddings, "Embeddings", "IconEmbed", [ModelTask.Embeddings], Runnable: true),
        new("time-series", "Time series", "IconStats", [ModelTask.TimeSeries], Runnable: false),
        new("3d-generation", "3D", "IconCube", [ModelTask.ThreeD], Runnable: false),
        new("bio", "Bio", "IconBio", [ModelTask.Bio], Runnable: false),
        new("robotics", "Robotics", "IconChip", [ModelTask.Robotics], Runnable: false),
        new(Other, "Other", "IconModels", [ModelTask.Other], Runnable: false),
    ];

    /// <summary>The presentation of a kind; a generic entry (titled by the kind, not runnable) for one that is not listed.</summary>
    /// <param name="kind">The kind of a model description.</param>
    public static ModelKindInfo For(string? kind)
    {
        foreach (var info in All)
        {
            if (string.Equals(info.Id, kind, StringComparison.OrdinalIgnoreCase))
            {
                return info;
            }
        }

        return string.IsNullOrEmpty(kind)
            ? For(Other)
            : new ModelKindInfo(kind, kind.Replace('-', ' '), "IconModels", [], Runnable: false);
    }

    /// <summary>The kind of the models of a Hub task.</summary>
    /// <param name="task">The task.</param>
    public static ModelKindInfo For(ModelTask task) => All.First(info => info.Tasks.Contains(task));
}
