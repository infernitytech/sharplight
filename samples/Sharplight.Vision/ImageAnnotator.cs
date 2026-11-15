using Sharplight.Imaging;

namespace Sharplight.Vision.Sample;

/// <summary>Draws boxes, masks and short labels into a copy of an RGB image and saves it as PNG.</summary>
public sealed class ImageAnnotator
{
    private static readonly (byte R, byte G, byte B)[] s_palette =
    [
        (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
        (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195),
    ];

    private readonly byte[] _rgb;
    private readonly int _width;
    private readonly int _height;

    /// <summary>Starts from a copy of <paramref name="image"/>.</summary>
    /// <param name="image">The image to draw on.</param>
    public ImageAnnotator(ImageInput image)
    {
        ArgumentNullException.ThrowIfNull(image);
        _width = image.Width;
        _height = image.Height;
        _rgb = (byte[])image.Rgb.Clone();
    }

    /// <summary>Draws a 2 pixel box in the color of <paramref name="colorIndex"/> with a label on a filled strip above it.</summary>
    /// <param name="box">The box in image pixels.</param>
    /// <param name="colorIndex">Selects the color (typically the class index).</param>
    /// <param name="label">The text (letters, digits and common punctuation; other characters are skipped).</param>
    public void DrawBox(BoundingBox box, int colorIndex, string label)
    {
        var color = s_palette[colorIndex % s_palette.Length];
        int x1 = (int)MathF.Round(box.X1);
        int y1 = (int)MathF.Round(box.Y1);
        int x2 = (int)MathF.Round(box.X2);
        int y2 = (int)MathF.Round(box.Y2);
        for (int t = 0; t < 2; t++)
        {
            FillRectangle(x1, y1 + t, x2, y1 + t + 1, color);
            FillRectangle(x1, y2 - t - 1, x2, y2 - t, color);
            FillRectangle(x1 + t, y1, x1 + t + 1, y2, color);
            FillRectangle(x2 - t - 1, y1, x2 - t, y2, color);
        }

        const int Scale = 2;
        int textWidth = label.Length * 6 * Scale;
        int top = Math.Max(0, y1 - 9 * Scale);
        FillRectangle(x1, top, x1 + textWidth + 2, top + 9 * Scale, color);
        DrawText(x1 + 2, top + Scale, label, Scale, (0, 0, 0));
    }

    /// <summary>Blends a color over the pixels where <paramref name="mask"/> is set.</summary>
    /// <param name="mask">One byte per pixel (row-major, image size); non-zero pixels are covered.</param>
    /// <param name="colorIndex">Selects the color.</param>
    /// <param name="opacity">The share of the color, 0..1.</param>
    public void DrawMask(ReadOnlySpan<byte> mask, int colorIndex, float opacity = 0.5f)
    {
        var color = s_palette[colorIndex % s_palette.Length];
        for (int i = 0; i < _width * _height; i++)
        {
            if (mask[i] != 0)
            {
                _rgb[3 * i] = Mix(_rgb[3 * i], color.R, opacity);
                _rgb[3 * i + 1] = Mix(_rgb[3 * i + 1], color.G, opacity);
                _rgb[3 * i + 2] = Mix(_rgb[3 * i + 2], color.B, opacity);
            }
        }
    }

    /// <summary>Draws a filled circle (a point prompt).</summary>
    /// <param name="cx">Center x.</param>
    /// <param name="cy">Center y.</param>
    /// <param name="radius">Radius in pixels.</param>
    /// <param name="colorIndex">Selects the color.</param>
    public void DrawPoint(int cx, int cy, int radius, int colorIndex)
    {
        var color = s_palette[colorIndex % s_palette.Length];
        for (int y = cy - radius; y <= cy + radius; y++)
        {
            for (int x = cx - radius; x <= cx + radius; x++)
            {
                if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= radius * radius)
                {
                    FillRectangle(x, y, x + 1, y + 1, color);
                }
            }
        }
    }

    /// <summary>Writes the image as PNG.</summary>
    /// <param name="path">The file to write.</param>
    public void Save(string path) => PngCodec.Write(path, new ImageInput(_width, _height, _rgb));

    private static byte Mix(byte background, byte color, float opacity) => (byte)Math.Clamp(background * (1 - opacity) + color * opacity, 0, 255);

    private void FillRectangle(int x1, int y1, int x2, int y2, (byte R, byte G, byte B) color)
    {
        for (int y = Math.Max(0, y1); y < Math.Min(_height, y2); y++)
        {
            for (int x = Math.Max(0, x1); x < Math.Min(_width, x2); x++)
            {
                int i = 3 * (y * _width + x);
                _rgb[i] = color.R;
                _rgb[i + 1] = color.G;
                _rgb[i + 2] = color.B;
            }
        }
    }

    private void DrawText(int x, int y, string text, int scale, (byte R, byte G, byte B) color)
    {
        foreach (char c in text.ToUpperInvariant())
        {
            if (TinyFont.Glyph(c) is { } rows)
            {
                for (int row = 0; row < 7; row++)
                {
                    for (int col = 0; col < 5; col++)
                    {
                        if ((rows[row] >> (4 - col) & 1) != 0)
                        {
                            FillRectangle(x + col * scale, y + row * scale, x + (col + 1) * scale, y + (row + 1) * scale, color);
                        }
                    }
                }
            }

            x += 6 * scale;
        }
    }
}
