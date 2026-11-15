namespace Sharplight.Vision;

/// <summary>An axis-aligned box in pixel coordinates (origin top left, x to the right, y down), by its corners.</summary>
/// <param name="X1">Left edge.</param>
/// <param name="Y1">Top edge.</param>
/// <param name="X2">Right edge.</param>
/// <param name="Y2">Bottom edge.</param>
public readonly record struct BoundingBox(float X1, float Y1, float X2, float Y2)
{
    /// <summary>Width (0 for an inverted box).</summary>
    public float Width => Math.Max(0, X2 - X1);

    /// <summary>Height (0 for an inverted box).</summary>
    public float Height => Math.Max(0, Y2 - Y1);

    /// <summary>The area.</summary>
    public float Area => Width * Height;

    /// <summary>The box of a center and size.</summary>
    /// <param name="cx">Center x.</param>
    /// <param name="cy">Center y.</param>
    /// <param name="width">Width.</param>
    /// <param name="height">Height.</param>
    public static BoundingBox FromCenter(float cx, float cy, float width, float height)
        => new(cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2);

    /// <summary>The box scaled by <paramref name="sx"/> horizontally and <paramref name="sy"/> vertically (normalized to pixel coordinates).</summary>
    /// <param name="sx">Horizontal factor.</param>
    /// <param name="sy">Vertical factor.</param>
    public BoundingBox Scale(float sx, float sy) => new(X1 * sx, Y1 * sy, X2 * sx, Y2 * sy);

    /// <summary>The box limited to an image of the given size.</summary>
    /// <param name="width">Image width.</param>
    /// <param name="height">Image height.</param>
    public BoundingBox Clip(float width, float height)
        => new(Math.Clamp(X1, 0, width), Math.Clamp(Y1, 0, height), Math.Clamp(X2, 0, width), Math.Clamp(Y2, 0, height));

    /// <summary>The intersection over union with <paramref name="other"/> (0 when both are empty).</summary>
    /// <param name="other">The other box.</param>
    public float IntersectionOverUnion(BoundingBox other)
    {
        float iw = Math.Min(X2, other.X2) - Math.Max(X1, other.X1);
        float ih = Math.Min(Y2, other.Y2) - Math.Max(Y1, other.Y1);
        if (iw <= 0 || ih <= 0)
        {
            return 0;
        }

        float intersection = iw * ih;
        float union = Area + other.Area - intersection;
        return union > 0 ? intersection / union : 0;
    }
}
