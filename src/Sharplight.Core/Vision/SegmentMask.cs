namespace Sharplight.Vision;

/// <summary>One candidate mask of an <see cref="ImageSegmentation"/>, at the size of the segmented image.</summary>
public sealed class SegmentMask
{
    /// <summary>Creates a mask.</summary>
    /// <param name="width">Width in pixels.</param>
    /// <param name="height">Height in pixels.</param>
    /// <param name="pixels">One entry per pixel (row-major), <see langword="true"/> inside the object.</param>
    /// <param name="score">The model's predicted IoU of the mask.</param>
    public SegmentMask(int width, int height, bool[] pixels, float score)
    {
        ArgumentNullException.ThrowIfNull(pixels);
        Width = width;
        Height = height;
        Pixels = pixels;
        Score = score;
        Area = pixels.Count(p => p);
    }

    /// <summary>Width in pixels.</summary>
    public int Width { get; }

    /// <summary>Height in pixels.</summary>
    public int Height { get; }

    /// <summary>One entry per pixel (row-major), <see langword="true"/> inside the object.</summary>
    public bool[] Pixels { get; }

    /// <summary>The model's predicted IoU of the mask (0 to 1).</summary>
    public float Score { get; }

    /// <summary>The number of pixels inside the object.</summary>
    public int Area { get; }
}
