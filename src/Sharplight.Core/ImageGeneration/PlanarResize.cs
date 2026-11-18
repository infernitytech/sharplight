using Sharplight.Vision;

namespace Sharplight.ImageGeneration;

/// <summary>Resizes planar float images [C, H, W] separably with antialiased bicubic weights (the image-to-image input path).</summary>
internal static class PlanarResize
{
    /// <summary>Resizes every channel of <paramref name="source"/> from <paramref name="inHeight"/>×<paramref name="inWidth"/>.</summary>
    /// <param name="source">The planes.</param>
    /// <param name="channels">The number of planes.</param>
    /// <param name="inHeight">The source height.</param>
    /// <param name="inWidth">The source width.</param>
    /// <param name="outHeight">The target height.</param>
    /// <param name="outWidth">The target width.</param>
    /// <param name="lanczos">Whether to resample with Lanczos weights (diffusers' <c>VaeImageProcessor</c> default) instead of bicubic ones.</param>
    public static float[] Resize(float[] source, int channels, int inHeight, int inWidth, int outHeight, int outWidth, bool lanczos = false)
    {
        if (inHeight == outHeight && inWidth == outWidth)
        {
            return source;
        }

        var rows = lanczos ? ImageResampling.AntialiasedLanczos(inHeight, outHeight) : ImageResampling.AntialiasedBicubic(inHeight, outHeight);
        var columns = lanczos ? ImageResampling.AntialiasedLanczos(inWidth, outWidth) : ImageResampling.AntialiasedBicubic(inWidth, outWidth);
        var result = new float[channels * outHeight * outWidth];
        var horizontal = new float[inHeight * outWidth];
        for (int c = 0; c < channels; c++)
        {
            int inPlane = c * inHeight * inWidth;
            Array.Clear(horizontal);
            for (int y = 0; y < inHeight; y++)
            {
                for (int ox = 0; ox < outWidth; ox++)
                {
                    float sum = 0;
                    for (int x = 0; x < inWidth; x++)
                    {
                        float weight = columns[ox * inWidth + x];
                        if (weight != 0)
                        {
                            sum += weight * source[inPlane + y * inWidth + x];
                        }
                    }

                    horizontal[y * outWidth + ox] = sum;
                }
            }

            int outPlane = c * outHeight * outWidth;
            for (int oy = 0; oy < outHeight; oy++)
            {
                for (int y = 0; y < inHeight; y++)
                {
                    float weight = rows[oy * inHeight + y];
                    if (weight == 0)
                    {
                        continue;
                    }

                    for (int ox = 0; ox < outWidth; ox++)
                    {
                        result[outPlane + oy * outWidth + ox] += weight * horizontal[y * outWidth + ox];
                    }
                }
            }
        }

        return result;
    }
}
