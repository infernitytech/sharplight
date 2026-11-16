namespace Sharplight.Studio.ViewModels.Vision;

/// <summary>What the vision page does with the chosen model; follows the model's kind.</summary>
internal enum VisionMode
{
    /// <summary>No model is chosen.</summary>
    None,

    /// <summary>Image classification: the best classes of the model's label set.</summary>
    Classify,

    /// <summary>Image embedding: the image's vector, or zero-shot scores of candidate labels when the model has a text tower.</summary>
    Embed,

    /// <summary>Object detection: boxes with labels and scores.</summary>
    Detect,

    /// <summary>Depth estimation: a depth map.</summary>
    Depth,

    /// <summary>Per-pixel segmentation: every pixel gets a class (colored overlay with a legend), or a matte cuts the background out.</summary>
    Semantic,

    /// <summary>Promptable segmentation: click points or draw a box, the mask of the object follows.</summary>
    Segment,

    /// <summary>Text recognition: the text of the image.</summary>
    Ocr,

    /// <summary>Upscaling: the image enlarged.</summary>
    Upscale,
}
