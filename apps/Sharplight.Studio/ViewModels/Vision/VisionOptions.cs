using Sharplight.Vision;

namespace Sharplight.Studio.ViewModels.Vision;

/// <summary>The vision page's parameters mapped to the task options of every mode (a mode uses its own).</summary>
/// <param name="Classification">Options of classification and zero-shot scoring.</param>
/// <param name="Detection">Options of object detection.</param>
/// <param name="Depth">Options of depth estimation.</param>
/// <param name="Embedding">Options of embedding.</param>
/// <param name="Template">The text a zero-shot label is put into (<c>{}</c> stands for the label).</param>
/// <param name="Colormap">The color map of the depth image (<see cref="Services.Imaging.DepthColormap"/>).</param>
/// <param name="Invert">Whether the depth colors are inverted.</param>
/// <param name="Segmentation">Options of segmentation.</param>
/// <param name="Recognition">Options of text recognition.</param>
/// <param name="Upscale">Options of upscaling.</param>
internal sealed record VisionOptions(ImageClassificationOptions Classification, ObjectDetectionOptions Detection, DepthEstimationOptions Depth,
    ImageEmbeddingOptions Embedding, string Template, string Colormap, bool Invert, SegmentationOptions Segmentation, TextRecognitionOptions Recognition, ImageUpscalingOptions Upscale);
