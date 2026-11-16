using Sharplight.Imaging;
using Sharplight.Vision;

namespace Sharplight.Studio.Services.Imaging;

/// <summary>Draws a per-pixel segmentation: every class as a translucent color over the image, and the cut-out of a matte (alpha channel) as a PNG.</summary>
internal static class SemanticOverlay
{
    private const float Opacity = 0.55f;

    /// <summary>A distinct color for a class, 0xRRGGBB (golden-ratio hues, so neighboring indices differ even with 150 classes).</summary>
    /// <param name="classId">The class index.</param>
    public static uint ColorOf(int classId)
    {
        double hue = (Math.Abs(classId) * 0.61803398875) % 1.0;
        double lightness = 0.45 + (0.15 * (Math.Abs(classId) % 3 / 2.0));
        double chroma = (1 - Math.Abs((2 * lightness) - 1)) * 0.85;
        double h = hue * 6;
        double x = chroma * (1 - Math.Abs((h % 2) - 1));
        (double r, double g, double b) = (int)h switch
        {
            0 => (chroma, x, 0.0),
            1 => (x, chroma, 0.0),
            2 => (0.0, chroma, x),
            3 => (0.0, x, chroma),
            4 => (x, 0.0, chroma),
            _ => (chroma, 0.0, x),
        };
        double m = lightness - (chroma / 2);
        return (uint)(((byte)((r + m) * 255) << 16) | ((byte)((g + m) * 255) << 8) | (byte)((b + m) * 255));
    }

    /// <summary>The image with every class in its color, as PNG; a matte model's result is the image cut out (transparent background) instead.</summary>
    /// <param name="image">The image the segmentation was computed for.</param>
    /// <param name="result">The segmentation, at the size of the image.</param>
    public static byte[] Render(ImageInput image, SemanticSegmentation result)
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(result);
        if (result.Width != image.Width || result.Height != image.Height)
        {
            throw new ArgumentException("The segmentation has to be at the size of the image.", nameof(result));
        }

        if (result.Alpha is { } matte)
        {
            var alpha = new byte[matte.Length];
            for (int i = 0; i < alpha.Length; i++)
            {
                alpha[i] = (byte)Math.Clamp((matte[i] * 255f) + 0.5f, 0, 255);
            }

            return PngCodec.EncodeRgba(image, alpha, System.IO.Compression.CompressionLevel.Fastest);
        }

        var rgb = (byte[])image.Rgb.Clone();
        var colors = new Dictionary<int, uint>();
        for (int i = 0; i < result.Labels.Length; i++)
        {
            int label = result.Labels[i];
            if (!colors.TryGetValue(label, out uint color))
            {
                color = colors[label] = ColorOf(label);
            }

            rgb[3 * i] = Mix(rgb[3 * i], (byte)(color >> 16));
            rgb[(3 * i) + 1] = Mix(rgb[(3 * i) + 1], (byte)(color >> 8));
            rgb[(3 * i) + 2] = Mix(rgb[(3 * i) + 2], (byte)color);
        }

        return ImageFiles.ToPng(new ImageInput(image.Width, image.Height, rgb));
    }

    private static byte Mix(byte source, byte color) => (byte)((source * (1 - Opacity)) + (color * Opacity) + 0.5f);
}
