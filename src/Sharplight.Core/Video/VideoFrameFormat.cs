namespace Sharplight.Video;

/// <summary>The pixel layout of a decoded <see cref="VideoFrame"/>.</summary>
public enum VideoFrameFormat
{
    /// <summary>Packed 8-bit RGB, 3 bytes per pixel, rows top to bottom without padding (the layout of <see cref="Vision.ImageInput"/>).</summary>
    Rgb24,

    /// <summary>
    /// Planar 8-bit YUV 4:2:0 (I420): the full-size Y plane, then the U and V planes of ((width + 1) / 2) x ((height + 1) / 2) samples,
    /// without padding; limited range with the source's matrix.
    /// </summary>
    Yuv420,
}
