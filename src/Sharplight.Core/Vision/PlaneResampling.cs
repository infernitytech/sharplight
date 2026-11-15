namespace Sharplight.Vision;

/// <summary>Host-side resampling of single-channel float maps (depth, masks, logits).</summary>
internal static class PlaneResampling
{
    /// <summary>Bilinear resize with half-pixel centers (PyTorch <c>interpolate(..., align_corners=False)</c>).</summary>
    /// <param name="source">The map, row-major.</param>
    /// <param name="sourceWidth">Its width.</param>
    /// <param name="sourceHeight">Its height.</param>
    /// <param name="width">The target width.</param>
    /// <param name="height">The target height.</param>
    public static float[] Bilinear(float[] source, int sourceWidth, int sourceHeight, int width, int height)
    {
        var result = new float[width * height];
        float sx = (float)sourceWidth / width;
        float sy = (float)sourceHeight / height;
        for (int y = 0; y < height; y++)
        {
            float fy = Math.Max(0, (y + 0.5f) * sy - 0.5f);
            int y0 = Math.Min((int)fy, sourceHeight - 1);
            int y1 = Math.Min(y0 + 1, sourceHeight - 1);
            float wy = fy - y0;
            for (int x = 0; x < width; x++)
            {
                float fx = Math.Max(0, (x + 0.5f) * sx - 0.5f);
                int x0 = Math.Min((int)fx, sourceWidth - 1);
                int x1 = Math.Min(x0 + 1, sourceWidth - 1);
                float wx = fx - x0;
                float top = source[y0 * sourceWidth + x0] * (1 - wx) + source[y0 * sourceWidth + x1] * wx;
                float bottom = source[y1 * sourceWidth + x0] * (1 - wx) + source[y1 * sourceWidth + x1] * wx;
                result[y * width + x] = top * (1 - wy) + bottom * wy;
            }
        }

        return result;
    }
}
