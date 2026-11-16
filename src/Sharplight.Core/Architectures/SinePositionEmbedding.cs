namespace Sharplight.Architectures;

/// <summary>The normalized 2-D sine position embedding of DETR-style detectors for an unpadded feature map.</summary>
internal static class SinePositionEmbedding
{
    /// <summary>
    /// The embedding [height · width, 2 · <paramref name="features"/>] (row-major over the grid; the first half of the channels encodes y, the second
    /// half x, each with interleaved sine and cosine at geometric wavelengths).
    /// </summary>
    /// <param name="height">Feature map height.</param>
    /// <param name="width">Feature map width.</param>
    /// <param name="features">Channels per axis.</param>
    /// <param name="temperature">The wavelength base (10000).</param>
    public static float[] Create(int height, int width, int features, double temperature = 10000)
    {
        const double Scale = 2 * Math.PI;
        const double Eps = 1e-6;
        var result = new float[height * width * 2 * features];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                double yEmbed = (y + 1) / (height + Eps) * Scale;
                double xEmbed = (x + 1) / (width + Eps) * Scale;
                int row = (y * width + x) * 2 * features;
                for (int i = 0; i < features; i++)
                {
                    double dim = Math.Pow(temperature, 2.0 * (i / 2) / features);
                    result[row + i] = (float)(i % 2 == 0 ? Math.Sin(yEmbed / dim) : Math.Cos(yEmbed / dim));
                    result[row + features + i] = (float)(i % 2 == 0 ? Math.Sin(xEmbed / dim) : Math.Cos(xEmbed / dim));
                }
            }
        }

        return result;
    }
}
