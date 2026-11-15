namespace Sharplight.Vision;

/// <summary>The result of <see cref="IDepthEstimator.EstimateAsync"/>: one value per pixel, row-major.</summary>
/// <param name="Width">Width in pixels.</param>
/// <param name="Height">Height in pixels.</param>
/// <param name="Values">The depth values (relative inverse depth for Depth Anything: larger is nearer), <c>Width · Height</c> entries.</param>
public sealed record DepthMap(int Width, int Height, float[] Values)
{
    /// <summary>The smallest value.</summary>
    public float Min => Values.Length > 0 ? Values.Min() : 0;

    /// <summary>The largest value.</summary>
    public float Max => Values.Length > 0 ? Values.Max() : 0;

    /// <summary>The map scaled to 0..255 (the smallest value to 0, the largest to 255), one byte per pixel.</summary>
    public byte[] ToGrayscale()
    {
        float min = Min;
        float range = Max - min;
        var result = new byte[Values.Length];
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = range > 0 ? (byte)Math.Clamp((Values[i] - min) / range * 255f + 0.5f, 0, 255) : (byte)0;
        }

        return result;
    }
}
