namespace Sharplight.ModelFeed;

/// <summary>Task names of <see cref="PackageModel.Tasks"/>: what a host may offer a model for.</summary>
public static class PackageTasks
{
    /// <summary>Chat completions.</summary>
    public const string Chat = "chat";

    /// <summary>Plain text completions.</summary>
    public const string Completion = "completion";

    /// <summary>Text embeddings.</summary>
    public const string Embedding = "embedding";

    /// <summary>Reranking (cross-encoder scores).</summary>
    public const string Rerank = "rerank";

    /// <summary>Speech to text.</summary>
    public const string Transcription = "transcription";

    /// <summary>Text to speech.</summary>
    public const string Speech = "speech";

    /// <summary>Voice activity detection.</summary>
    public const string VoiceActivity = "voice-activity";

    /// <summary>Image generation.</summary>
    public const string ImageGeneration = "image-generation";

    /// <summary>Video generation.</summary>
    public const string VideoGeneration = "video-generation";

    /// <summary>Image embeddings.</summary>
    public const string ImageEmbedding = "image-embedding";

    /// <summary>Image classification.</summary>
    public const string ImageClassification = "image-classification";

    /// <summary>Object detection.</summary>
    public const string ObjectDetection = "object-detection";

    /// <summary>Image segmentation.</summary>
    public const string ImageSegmentation = "image-segmentation";

    /// <summary>Depth estimation.</summary>
    public const string DepthEstimation = "depth-estimation";

    /// <summary>Text recognition in images.</summary>
    public const string Ocr = "ocr";

    /// <summary>Image upscaling.</summary>
    public const string ImageUpscaling = "image-upscaling";

    /// <summary>Audio encoding and decoding.</summary>
    public const string AudioCodec = "audio-codec";

    /// <summary>Video (and still image) restoration and super-resolution.</summary>
    public const string VideoEnhancement = "video-enhancement";
}
