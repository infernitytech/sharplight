using Sharplight.Vision;

namespace Sharplight.Studio.ViewModels.Video;

/// <summary>One frame of the Enhance page's comparison: the input frame and the enhanced frame of the same index, both at the preview size so they overlay exactly.</summary>
/// <param name="Index">The frame's 0-based index in the clip.</param>
/// <param name="Original">The input frame, scaled (bilinear) to the preview size.</param>
/// <param name="Enhanced">The enhanced frame at the preview size (the model's output itself unless the memory cap reduced it).</param>
internal sealed record ComparisonFrame(int Index, ImageInput Original, ImageInput Enhanced)
{
    /// <summary>The bytes the two frames hold.</summary>
    public long Bytes => (long)Original.Rgb.Length + Enhanced.Rgb.Length;
}
