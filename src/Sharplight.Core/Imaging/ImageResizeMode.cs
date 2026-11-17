namespace Sharplight.Imaging;

/// <summary>How an <see cref="ImageProcessor"/> brings an image to the model's input size (before the optional center crop).</summary>
public enum ImageResizeMode
{
    /// <summary>No resizing.</summary>
    None,

    /// <summary>Resize to exactly <see cref="ImageProcessorConfig.Width"/> × <see cref="ImageProcessorConfig.Height"/>, ignoring the aspect ratio.</summary>
    Exact,

    /// <summary>Scale so the shorter edge becomes <see cref="ImageProcessorConfig.ShortestEdge"/>, keeping the aspect ratio.</summary>
    ShortestEdge,

    /// <summary>Scale so the longer edge becomes <see cref="ImageProcessorConfig.LongestEdge"/>, keeping the aspect ratio.</summary>
    LongestEdge,

    /// <summary>
    /// Scale to the largest size that fits inside <see cref="ImageProcessorConfig.Width"/> × <see cref="ImageProcessorConfig.Height"/> with the aspect ratio kept (sizes rounded to the nearest
    /// integer, ties to even); with the center crop set to the same size the rest is padded with <see cref="ImageProcessorConfig.PadColor"/> (letterboxing).
    /// </summary>
    Contain,
}
