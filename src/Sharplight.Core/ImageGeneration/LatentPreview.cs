using Sharplight.Vision;

namespace Sharplight.ImageGeneration;

/// <summary>
/// Cheap previews of a latent image: a per-pixel linear map from latent channels to RGB (the model's latent RGB factors), at latent
/// resolution, without running the VAE.
/// </summary>
internal static class LatentPreview
{
    /// <summary>Renders <paramref name="latent"/> [C, h, w] with the [C, 3] row-major <paramref name="factors"/> (output in [-1, 1]).</summary>
    public static ImageInput Render(ReadOnlySpan<float> latent, int channels, int height, int width, float[] factors)
    {
        int pixels = height * width;
        var rgb = new byte[pixels * 3];
        for (int p = 0; p < pixels; p++)
        {
            for (int color = 0; color < 3; color++)
            {
                float value = 0;
                for (int c = 0; c < channels; c++)
                {
                    value += latent[c * pixels + p] * factors[c * 3 + color];
                }

                rgb[p * 3 + color] = (byte)Math.Clamp(MathF.Round((value + 1f) * 127.5f), 0f, 255f);
            }
        }

        return new ImageInput(width, height, rgb);
    }
}
