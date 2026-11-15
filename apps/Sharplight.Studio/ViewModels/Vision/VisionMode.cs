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
}
