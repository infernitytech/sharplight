using Sharplight.Vision;

namespace Sharplight.Video.Tests;

/// <summary>Synthetic clips for the encoder tests.</summary>
internal static class TestClips
{
    /// <summary>A smooth diagonal color gradient that moves by a few pixels per frame.</summary>
    public static VideoClip MovingGradient(int width = 64, int height = 48, int frames = 10, double fps = 10)
    {
        var list = new List<ImageInput>();
        for (int f = 0; f < frames; f++)
        {
            var rgb = new byte[width * height * 3];
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int o = (y * width + x) * 3;
                    rgb[o] = (byte)((x * 4 + f * 6) % 256);
                    rgb[o + 1] = (byte)((y * 5 + f * 3) % 256);
                    rgb[o + 2] = (byte)(128 + (x + y) % 100);
                }
            }

            list.Add(new ImageInput(width, height, rgb));
        }

        return new VideoClip(list, fps);
    }

    /// <summary>A few flat colors with a moving square: exactly representable in a GIF palette, and most of each frame is unchanged.</summary>
    public static VideoClip MovingSquare(int width = 40, int height = 30, int frames = 8, double fps = 8)
    {
        var list = new List<ImageInput>();
        for (int f = 0; f < frames; f++)
        {
            var rgb = new byte[width * height * 3];
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int o = (y * width + x) * 3;
                    bool inside = x >= f * 3 && x < f * 3 + 8 && y >= 5 && y < 13;
                    rgb[o] = inside ? (byte)250 : (byte)(x < 20 ? 10 : 40);
                    rgb[o + 1] = inside ? (byte)(20 + f * 20) : (byte)(y < 15 ? 90 : 160);
                    rgb[o + 2] = inside ? (byte)30 : (byte)200;
                }
            }

            list.Add(new ImageInput(width, height, rgb));
        }

        return new VideoClip(list, fps);
    }

    /// <summary>Peak signal-to-noise ratio between two RGB frames in dB.</summary>
    public static double Psnr(ReadOnlySpan<byte> a, ReadOnlySpan<byte> b)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = a[i] - b[i];
            sum += d * d;
        }

        return sum == 0 ? double.PositiveInfinity : 10 * Math.Log10(255.0 * 255.0 * a.Length / sum);
    }

    /// <summary>A unique path in the temp directory.</summary>
    public static string TempPath(string extension) => Path.Combine(Path.GetTempPath(), $"sl-video-{Guid.NewGuid():N}{extension}");
}
