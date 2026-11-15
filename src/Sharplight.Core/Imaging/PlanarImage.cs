using Sharplight.Vision;

namespace Sharplight.Imaging;

/// <summary>
/// Conversions between 8-bit interleaved RGB images (<see cref="ImageInput"/>) and the planar float tensors image models consume and
/// produce (<c>[3, height, width]</c>, channel-major), e.g. a VAE decoder's output in [-1, 1] or a VAE encoder's input.
/// </summary>
public static class PlanarImage
{
    /// <summary>Converts a planar RGB tensor to an image, mapping <paramref name="low"/>..<paramref name="high"/> to 0..255 (clamped, rounded).</summary>
    /// <param name="planes">The <c>3 · height · width</c> values, channel-major.</param>
    /// <param name="width">Width in pixels (positive).</param>
    /// <param name="height">Height in pixels (positive).</param>
    /// <param name="low">The value that maps to 0.</param>
    /// <param name="high">The value that maps to 255.</param>
    /// <exception cref="ArgumentException">The size is not positive or the tensor has the wrong length.</exception>
    public static ImageInput FromPlanar(ReadOnlySpan<float> planes, int width, int height, float low = -1f, float high = 1f)
    {
        int pixels = CheckSize(planes.Length, width, height);
        var rgb = new byte[pixels * 3];
        float scale = 255f / (high - low);
        for (int channel = 0; channel < 3; channel++)
        {
            var plane = planes.Slice(channel * pixels, pixels);
            for (int i = 0; i < pixels; i++)
            {
                rgb[i * 3 + channel] = (byte)Math.Clamp(MathF.Round((plane[i] - low) * scale), 0f, 255f);
            }
        }

        return new ImageInput(width, height, rgb);
    }

    /// <summary>Converts an image to a planar RGB tensor, mapping 0..255 to <paramref name="low"/>..<paramref name="high"/>.</summary>
    /// <param name="image">The image.</param>
    /// <param name="planes">Receives the <c>3 · height · width</c> values, channel-major.</param>
    /// <param name="low">The value 0 maps to.</param>
    /// <param name="high">The value 255 maps to.</param>
    /// <exception cref="ArgumentException">The image is invalid or <paramref name="planes"/> has the wrong length.</exception>
    public static void ToPlanar(ImageInput image, Span<float> planes, float low = -1f, float high = 1f)
    {
        image.Validate();
        int pixels = CheckSize(planes.Length, image.Width, image.Height);
        float scale = (high - low) / 255f;
        for (int channel = 0; channel < 3; channel++)
        {
            var plane = planes.Slice(channel * pixels, pixels);
            for (int i = 0; i < pixels; i++)
            {
                plane[i] = low + image.Rgb[i * 3 + channel] * scale;
            }
        }
    }

    private static int CheckSize(int length, int width, int height)
    {
        if (width <= 0 || height <= 0)
        {
            throw new ArgumentException($"Image size {width}x{height} has no pixels.");
        }

        long pixels = (long)width * height;
        if (length != 3 * pixels)
        {
            throw new ArgumentException($"Expected {3 * pixels} planar values for {width}x{height}, got {length}.");
        }

        return (int)pixels;
    }
}
