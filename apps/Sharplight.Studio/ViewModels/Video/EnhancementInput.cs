using Sharplight.Vision;

namespace Sharplight.Studio.ViewModels.Video;

/// <summary>The input chosen on the Enhance page: a still image (decoded) or a video file (probed; its frames are read when the run starts).</summary>
/// <param name="Path">The file.</param>
/// <param name="Image">The decoded image, or null for a video.</param>
/// <param name="Width">Width of the frames in pixels.</param>
/// <param name="Height">Height of the frames in pixels.</param>
/// <param name="Frames">Frames in the file (1 for an image).</param>
/// <param name="FramesPerSecond">The playback rate of a video, 0 for an image.</param>
internal sealed record EnhancementInput(string Path, ImageInput? Image, int Width, int Height, long Frames, double FramesPerSecond)
{
    /// <summary>The input is a video (its frames are decoded when the run starts).</summary>
    public bool IsVideo => Image is null;
}
