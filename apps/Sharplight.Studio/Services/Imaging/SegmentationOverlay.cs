using Sharplight.Vision;

namespace Sharplight.Studio.Services.Imaging;

/// <summary>Draws a segmentation on a copy of the image: the mask as a translucent color, the prompt's points as dots (green: on the object, red: not) and its box as an outline.</summary>
internal static class SegmentationOverlay
{
    private const float Opacity = 0.5f;
    private static readonly (byte R, byte G, byte B) s_maskColor = (30, 144, 255);
    private static readonly (byte R, byte G, byte B) s_positive = (40, 200, 70);
    private static readonly (byte R, byte G, byte B) s_negative = (230, 40, 50);
    private static readonly (byte R, byte G, byte B) s_box = (255, 215, 0);

    /// <summary>Renders the overlay.</summary>
    /// <param name="image">The image.</param>
    /// <param name="mask">The mask to show (at the size of the image), or <see langword="null"/> for none.</param>
    /// <param name="points">The prompt's points.</param>
    /// <param name="box">The prompt's box, or <see langword="null"/>.</param>
    public static ImageInput Render(ImageInput image, SegmentMask? mask, IEnumerable<SegmentationPoint> points, BoundingBox? box)
    {
        ArgumentNullException.ThrowIfNull(image);
        var rgb = (byte[])image.Rgb.Clone();
        if (mask is not null && mask.Width == image.Width && mask.Height == image.Height)
        {
            for (int i = 0; i < mask.Pixels.Length; i++)
            {
                if (!mask.Pixels[i])
                {
                    continue;
                }

                rgb[3 * i] = Mix(rgb[3 * i], s_maskColor.R);
                rgb[(3 * i) + 1] = Mix(rgb[(3 * i) + 1], s_maskColor.G);
                rgb[(3 * i) + 2] = Mix(rgb[(3 * i) + 2], s_maskColor.B);
            }
        }

        int scale = Math.Max(1, Math.Max(image.Width, image.Height) / 480);
        if (box is { } outline)
        {
            Outline(rgb, image.Width, image.Height, outline, Math.Max(2, scale + 1));
        }

        foreach (var point in points)
        {
            Dot(rgb, image.Width, image.Height, (int)MathF.Round(point.X), (int)MathF.Round(point.Y), (3 * scale) + 1, point.IsPositive ? s_positive : s_negative);
        }

        return new ImageInput(image.Width, image.Height, rgb);
    }

    private static byte Mix(byte source, byte color) => (byte)((source * (1 - Opacity)) + (color * Opacity) + 0.5f);

    private static void Outline(byte[] rgb, int width, int height, BoundingBox box, int line)
    {
        int x1 = (int)MathF.Round(box.X1);
        int y1 = (int)MathF.Round(box.Y1);
        int x2 = (int)MathF.Round(box.X2);
        int y2 = (int)MathF.Round(box.Y2);
        for (int t = 0; t < line; t++)
        {
            Fill(rgb, width, height, x1, y1 + t, x2, y1 + t + 1, s_box);
            Fill(rgb, width, height, x1, y2 - t - 1, x2, y2 - t, s_box);
            Fill(rgb, width, height, x1 + t, y1, x1 + t + 1, y2, s_box);
            Fill(rgb, width, height, x2 - t - 1, y1, x2 - t, y2, s_box);
        }
    }

    private static void Fill(byte[] rgb, int width, int height, int x1, int y1, int x2, int y2, (byte R, byte G, byte B) color)
    {
        for (int y = Math.Max(0, y1); y < Math.Min(height, y2); y++)
        {
            for (int x = Math.Max(0, x1); x < Math.Min(width, x2); x++)
            {
                int i = 3 * ((y * width) + x);
                rgb[i] = color.R;
                rgb[i + 1] = color.G;
                rgb[i + 2] = color.B;
            }
        }
    }

    private static void Dot(byte[] rgb, int width, int height, int cx, int cy, int radius, (byte R, byte G, byte B) color)
    {
        for (int y = Math.Max(0, cy - radius - 1); y <= Math.Min(height - 1, cy + radius + 1); y++)
        {
            for (int x = Math.Max(0, cx - radius - 1); x <= Math.Min(width - 1, cx + radius + 1); x++)
            {
                int distance = ((x - cx) * (x - cx)) + ((y - cy) * (y - cy));
                if (distance > (radius + 1) * (radius + 1))
                {
                    continue;
                }

                int i = 3 * ((y * width) + x);
                var fill = distance > radius * radius ? ((byte)255, (byte)255, (byte)255) : color;
                rgb[i] = fill.Item1;
                rgb[i + 1] = fill.Item2;
                rgb[i + 2] = fill.Item3;
            }
        }
    }
}
