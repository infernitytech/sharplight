namespace Sharplight.Hub;

/// <summary>What a Hub repository's model does, grouped the way Discover filters them (see <see cref="ModelTasks"/>).</summary>
public enum ModelTask
{
    /// <summary>Language models that continue or answer text (<c>text-generation</c>, <c>text2text-generation</c>, ...).</summary>
    TextGeneration,

    /// <summary>Language models that also read images or video (<c>image-text-to-text</c>, <c>visual-question-answering</c>, ...).</summary>
    VisionLanguage,

    /// <summary>Speech recognition (<c>automatic-speech-recognition</c>).</summary>
    SpeechToText,

    /// <summary>Speech synthesis (<c>text-to-speech</c>, <c>text-to-audio</c>).</summary>
    TextToSpeech,

    /// <summary>Images from text or images (<c>text-to-image</c>, <c>image-to-image</c>, ...).</summary>
    ImageGeneration,

    /// <summary>Video from text or images (<c>text-to-video</c>, <c>image-to-video</c>).</summary>
    VideoGeneration,

    /// <summary>Image and video classification.</summary>
    ImageClassification,

    /// <summary>Object detection.</summary>
    ObjectDetection,

    /// <summary>Image segmentation.</summary>
    Segmentation,

    /// <summary>Depth estimation.</summary>
    DepthEstimation,

    /// <summary>Optical character recognition and document understanding (<c>image-to-text</c>, <c>document-question-answering</c>).</summary>
    Ocr,

    /// <summary>Image super-resolution.</summary>
    Upscaling,

    /// <summary>Audio that is neither speech recognition nor synthesis (classification, audio-to-audio, voice activity).</summary>
    Audio,

    /// <summary>Embedding, retrieval and ranking (<c>feature-extraction</c>, <c>sentence-similarity</c>, <c>text-ranking</c>).</summary>
    Embeddings,

    /// <summary>Time-series forecasting.</summary>
    TimeSeries,

    /// <summary>3D generation and reconstruction (<c>text-to-3d</c>, <c>image-to-3d</c>).</summary>
    ThreeD,

    /// <summary>Biology and chemistry models (protein, molecule and genomics models).</summary>
    Bio,

    /// <summary>Robotics policies.</summary>
    Robotics,

    /// <summary>Other tasks (tabular, reinforcement learning, classic NLP tasks) and repositories without a task.</summary>
    Other,
}
