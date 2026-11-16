using Sharplight.Vision;

namespace Sharplight.Video;

/// <summary>Converts interleaved RGB to planar I420 (4:2:0) with BT.601 limited-range coefficients, the default assumption of players.</summary>
internal static class RgbToI420
{
    /// <summary>Allocates the three planes for a frame size: Y is width x height, U and V are ceil(width/2) x ceil(height/2).</summary>
    public static (byte[] Y, byte[] U, byte[] V) AllocatePlanes(int width, int height)
    {
        int chroma = ((width + 1) / 2) * ((height + 1) / 2);
        return (new byte[width * height], new byte[chroma], new byte[chroma]);
    }

    /// <summary>Converts one frame into planes from <see cref="AllocatePlanes"/>; chroma is the average of each 2x2 block (edge pixels repeat).</summary>
    public static void Convert(ImageInput image, byte[] y, byte[] u, byte[] v)
    {
        int width = image.Width;
        int height = image.Height;
        ReadOnlySpan<byte> rgb = image.Rgb;
        for (int row = 0; row < height; row++)
        {
            for (int col = 0; col < width; col++)
            {
                int offset = (row * width + col) * 3;
                y[row * width + col] = (byte)(16 + ((66 * rgb[offset] + 129 * rgb[offset + 1] + 25 * rgb[offset + 2] + 128) >> 8));
            }
        }

        int chromaWidth = (width + 1) / 2;
        int chromaHeight = (height + 1) / 2;
        for (int row = 0; row < chromaHeight; row++)
        {
            for (int col = 0; col < chromaWidth; col++)
            {
                int x0 = col * 2;
                int x1 = Math.Min(x0 + 1, width - 1);
                int y0 = row * 2;
                int y1 = Math.Min(y0 + 1, height - 1);
                int r = 0;
                int g = 0;
                int b = 0;
                foreach (int py in (ReadOnlySpan<int>)[y0, y1])
                {
                    foreach (int px in (ReadOnlySpan<int>)[x0, x1])
                    {
                        int offset = (py * width + px) * 3;
                        r += rgb[offset];
                        g += rgb[offset + 1];
                        b += rgb[offset + 2];
                    }
                }

                // The sums are four times the average, so the shift is two bits larger than for one pixel.
                u[row * chromaWidth + col] = Clamp(128 + ((-38 * r - 74 * g + 112 * b + 512) >> 10));
                v[row * chromaWidth + col] = Clamp(128 + ((112 * r - 94 * g - 18 * b + 512) >> 10));
            }
        }
    }

    private static byte Clamp(int value) => (byte)Math.Clamp(value, 0, 255);
}
