using Sharplight.Vision;

namespace Sharplight.ImageGeneration;

/// <summary>Resizes planar float images [C, H, W] separably with antialiased bicubic weights (the image-to-image input path).</summary>
internal static class PlanarResize
{
    /// <summary>Resizes every channel of <paramref name="source"/> from <paramref name="inHeight"/>×<paramref name="inWidth"/>.</summary>
    public static float[] Resize(float[] source, int channels, int inHeight, int inWidth, int outHeight, int outWidth)
    {
        if (inHeight == outHeight && inWidth == outWidth)
        {
            return source;
        }

        var rows = ImageResampling.AntialiasedBicubic(inHeight, outHeight);
        var columns = ImageResampling.AntialiasedBicubic(inWidth, outWidth);
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
