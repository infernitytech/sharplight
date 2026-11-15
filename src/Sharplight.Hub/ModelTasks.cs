using System.Collections.Frozen;

namespace Sharplight.Hub;

/// <summary>
/// Maps the Hub's pipeline tags and library names to <see cref="ModelTask"/> groups, and back to the pipeline tags a search sends.
/// A repository without a pipeline tag is classified by its library (<c>diffusers</c>, <c>whisper.cpp</c>, ...), else as
/// <see cref="ModelTask.Other"/>. Two Hub tags are too coarse for one task, so the repository name refines them: image-to-image
/// models named like upscalers are <see cref="ModelTask.Upscaling"/>, image-to-text and vision-language models named
/// like OCR models are <see cref="ModelTask.Ocr"/>.
/// </summary>
public static class ModelTasks
{
    private static readonly (ModelTask Task, string[] Pipelines)[] s_pipelines =
    [
        (ModelTask.TextGeneration, ["text-generation", "text2text-generation", "fill-mask", "summarization", "translation", "conversational"]),
        (ModelTask.VisionLanguage, ["image-text-to-text", "visual-question-answering", "video-text-to-text", "any-to-any"]),
        (ModelTask.SpeechToText, ["automatic-speech-recognition"]),
        (ModelTask.TextToSpeech, ["text-to-speech", "text-to-audio"]),
        (ModelTask.ImageGeneration, ["text-to-image", "image-to-image", "unconditional-image-generation"]),
        (ModelTask.VideoGeneration, ["text-to-video", "image-to-video", "video-to-video"]),
        (ModelTask.ImageClassification, ["image-classification", "video-classification", "zero-shot-image-classification"]),
        (ModelTask.ObjectDetection, ["object-detection", "zero-shot-object-detection"]),
        (ModelTask.Segmentation, ["image-segmentation", "mask-generation", "keypoint-detection"]),
        (ModelTask.DepthEstimation, ["depth-estimation"]),
        (ModelTask.Ocr, ["image-to-text", "document-question-answering", "visual-document-retrieval"]),
        (ModelTask.Upscaling, []),
        (ModelTask.Audio, ["audio-classification", "audio-to-audio", "voice-activity-detection", "zero-shot-audio-classification"]),
        (ModelTask.Embeddings, ["feature-extraction", "sentence-similarity", "text-ranking", "image-feature-extraction"]),
        (ModelTask.TimeSeries, ["time-series-forecasting"]),
        (ModelTask.ThreeD, ["text-to-3d", "image-to-3d"]),
        (ModelTask.Bio, []),
        (ModelTask.Robotics, ["robotics"]),
    ];

    private static readonly string[] s_upscalerNames = ["upscal", "esrgan", "super-resolution", "superresolution", "swinir", "supir"];
    private static readonly string[] s_ocrNames = ["ocr", "trocr", "donut", "nougat", "paddleocr", "got-ocr"];
    private static readonly string[] s_bioNames = ["protein", "esm2", "esmfold", "alphafold", "chemberta", "molecul", "genomic", "nucleotide", "biogpt"];

    private static readonly FrozenDictionary<string, ModelTask> s_byPipeline = s_pipelines
        .SelectMany(entry => entry.Pipelines.Select(pipeline => (pipeline, entry.Task)))
        .ToFrozenDictionary(pair => pair.pipeline, pair => pair.Task, StringComparer.OrdinalIgnoreCase);

    private static readonly FrozenDictionary<string, ModelTask> s_byLibrary = new Dictionary<string, ModelTask>(StringComparer.OrdinalIgnoreCase)
    {
        ["diffusers"] = ModelTask.ImageGeneration,
        ["whisper.cpp"] = ModelTask.SpeechToText,
        ["faster-whisper"] = ModelTask.SpeechToText,
        ["sentence-transformers"] = ModelTask.Embeddings,
        ["timesfm"] = ModelTask.TimeSeries,
        ["chronos"] = ModelTask.TimeSeries,
        ["lerobot"] = ModelTask.Robotics,
        ["biopython"] = ModelTask.Bio,
    }.ToFrozenDictionary(StringComparer.OrdinalIgnoreCase);

    /// <summary>Every task, in the order Discover lists them.</summary>
    public static IReadOnlyList<ModelTask> All { get; } = Enum.GetValues<ModelTask>();

    /// <summary>The task of a repository.</summary>
    /// <param name="pipeline">The pipeline tag (<c>text-generation</c>), or <see langword="null"/>.</param>
    /// <param name="library">The library name (<c>diffusers</c>), or <see langword="null"/>.</param>
    /// <param name="repository">The repository id, which refines the coarse pipeline tags, or <see langword="null"/>.</param>
    public static ModelTask From(string? pipeline, string? library, string? repository = null)
    {
        var task = pipeline is { Length: > 0 } && s_byPipeline.TryGetValue(pipeline, out var fromPipeline)
            ? fromPipeline
            : library is { Length: > 0 } && s_byLibrary.TryGetValue(library, out var fromLibrary) ? fromLibrary : ModelTask.Other;
        if (repository is not { Length: > 0 })
        {
            return task;
        }

        return task switch
        {
            ModelTask.ImageGeneration when pipeline is "image-to-image" && Names(repository, s_upscalerNames) => ModelTask.Upscaling,
            ModelTask.VisionLanguage or ModelTask.Ocr when Names(repository, s_ocrNames) => ModelTask.Ocr,
            ModelTask.Other or ModelTask.Embeddings when Names(repository, s_bioNames) => ModelTask.Bio,
            _ => task,
        };
    }

    /// <summary>The pipeline tags the Hub knows for a task; empty for tasks the Hub has no tag for (<see cref="ModelTask.Upscaling"/>, <see cref="ModelTask.Bio"/>, <see cref="ModelTask.Other"/>).</summary>
    /// <param name="task">The task.</param>
    public static IReadOnlyList<string> PipelineTags(ModelTask task)
    {
        foreach (var entry in s_pipelines)
        {
            if (entry.Task == task)
            {
                return entry.Pipelines;
            }
        }

        return [];
    }

    /// <summary>
    /// The pipeline tags a search filters by: <see cref="PipelineTags"/>, and <c>image-to-image</c> for <see cref="ModelTask.Upscaling"/>
    /// (which has no tag of its own; the repository name tells upscalers apart).
    /// </summary>
    /// <param name="task">The task.</param>
    public static IReadOnlyList<string> SearchTags(ModelTask task) => task == ModelTask.Upscaling ? ["image-to-image"] : PipelineTags(task);

    /// <summary>True for tasks whose repositories are told apart by name, so a search asks for more repositories than it returns.</summary>
    /// <param name="task">The task.</param>
    public static bool IsRefinedByName(ModelTask task) => task is ModelTask.Upscaling or ModelTask.ImageGeneration or ModelTask.Ocr or ModelTask.VisionLanguage
        or ModelTask.Bio or ModelTask.Embeddings or ModelTask.Other;

    /// <summary>The search text that finds the models of a task whose pipeline tags are too coarse (upscalers, OCR, biology), or empty.</summary>
    /// <param name="task">The task.</param>
    public static string SearchHint(ModelTask task) => task switch
    {
        ModelTask.Upscaling => "upscale",
        ModelTask.Bio => "protein",
        _ => "",
    };

    private static bool Names(string repository, string[] hints)
    {
        var name = repository[(repository.LastIndexOf('/') + 1)..];
        foreach (var hint in hints)
        {
            if (name.Contains(hint, StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }

        return false;
    }
}
