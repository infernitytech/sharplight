using Sharplight.Audio;
using Sharplight.Vision;

namespace Sharplight.Video.Tests;

/// <summary>Synthetic clips for the encoder tests.</summary>
internal static class TestClips
{
    /// <summary>A smooth color gradient (no sharp edges, which 4:2:0 chroma cannot represent) that shifts a little every frame.</summary>
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
                    rgb[o] = (byte)Math.Min(255, 20 + x * 2 + f * 5);
                    rgb[o + 1] = (byte)Math.Min(255, 30 + y * 3 + f * 2);
                    rgb[o + 2] = (byte)Math.Min(255, 100 + x + y + f);
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

    /// <summary>A stereo sine tone at half amplitude.</summary>
    public static AudioBuffer Tone(double frequency, double seconds, int rate)
    {
        int count = (int)(seconds * rate);
        float[] samples = new float[count * 2];
        for (int i = 0; i < count; i++)
        {
            float value = (float)(0.5 * Math.Sin(2 * Math.PI * frequency * i / rate));
            samples[2 * i] = value;
            samples[2 * i + 1] = value;
        }

        return new AudioBuffer(samples, rate, 2);
    }

    /// <summary>The RMS of the samples between two times in seconds.</summary>
    public static double Rms(AudioBuffer audio, double from, double to)
    {
        int first = (int)(from * audio.SampleRate) * audio.Channels;
        int last = Math.Min(audio.Samples.Length, (int)(to * audio.SampleRate) * audio.Channels);
        double sum = 0;
        for (int i = first; i < last; i++)
        {
            sum += audio.Samples[i] * (double)audio.Samples[i];
        }

        return Math.Sqrt(sum / Math.Max(1, last - first));
    }
}
