using Sharplight.Vision;

namespace Sharplight.Studio.Services.Imaging;

/// <summary>
/// Resizes 8-bit RGB frames for display: plain bilinear (half-pixel centres) when enlarging, the same triangle filter widened to the
/// shrink factor when reducing (so a smaller preview averages its source instead of skipping pixels). Separable, rows in parallel.
/// </summary>
internal static class FrameResampler
{
    private const int Channels = 3;

    /// <summary>The frame at <paramref name="width"/> × <paramref name="height"/>; the frame itself when it already has that size.</summary>
    /// <param name="source">The frame.</param>
    /// <param name="width">Target width in pixels (positive).</param>
    /// <param name="height">Target height in pixels (positive).</param>
    public static ImageInput Resize(ImageInput source, int width, int height)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        if (source.Width == width && source.Height == height)
        {
            return source;
        }

        var horizontal = Weights.For(source.Width, width);
        var vertical = Weights.For(source.Height, height);

        // Horizontal pass into a float buffer of source height × target width, then the vertical pass into bytes.
        var rows = new float[source.Height * width * Channels];
        byte[] input = source.Rgb;
        Parallel.For(0, source.Height, y =>
        {
            int sourceRow = y * source.Width * Channels;
            int targetRow = y * width * Channels;
            for (int x = 0; x < width; x++)
            {
                float r = 0;
                float g = 0;
                float b = 0;
                int first = horizontal.Start[x];
                for (int k = 0; k < horizontal.Count[x]; k++)
                {
                    float weight = horizontal.Values[horizontal.Offset[x] + k];
                    int at = sourceRow + (first + k) * Channels;
                    r += input[at] * weight;
                    g += input[at + 1] * weight;
                    b += input[at + 2] * weight;
                }

                int to = targetRow + x * Channels;
                rows[to] = r;
                rows[to + 1] = g;
                rows[to + 2] = b;
            }
        });

        var output = new byte[width * height * Channels];
        int stride = width * Channels;
        Parallel.For(0, height, y =>
        {
            int first = vertical.Start[y];
            int targetRow = y * stride;
            var sum = new float[stride];
            for (int k = 0; k < vertical.Count[y]; k++)
            {
                float weight = vertical.Values[vertical.Offset[y] + k];
                int sourceRow = (first + k) * stride;
                for (int i = 0; i < stride; i++)
                {
                    sum[i] += rows[sourceRow + i] * weight;
                }
            }

            for (int i = 0; i < stride; i++)
            {
                output[targetRow + i] = (byte)Math.Clamp(sum[i] + 0.5f, 0, 255);
            }
        });
        return new ImageInput(width, height, output);
    }

    /// <summary>The normalized triangle-filter taps of each output position along one axis.</summary>
    private sealed record Weights(int[] Start, int[] Count, int[] Offset, float[] Values)
    {
        public static Weights For(int inSize, int outSize)
        {
            double scale = (double)inSize / outSize;
            double support = Math.Max(1, scale);
            var start = new int[outSize];
            var count = new int[outSize];
            var offset = new int[outSize];
            var values = new List<float>();
            for (int o = 0; o < outSize; o++)
            {
                double center = (o + 0.5) * scale - 0.5;
                int from = Math.Max(0, (int)Math.Floor(center - support + 1));
                int to = Math.Min(inSize - 1, (int)Math.Ceiling(center + support - 1));
                if (to < from)
                {
                    from = to = Math.Clamp((int)Math.Round(center), 0, inSize - 1);
                }

                double total = 0;
                var taps = new double[to - from + 1];
                for (int i = from; i <= to; i++)
                {
                    double weight = Math.Max(0, 1 - Math.Abs(i - center) / support);
                    taps[i - from] = weight;
                    total += weight;
                }

                if (total <= 0)
                {
                    // The centre sits outside the source (edge of an enlargement): take the nearest pixel.
                    int nearest = Math.Clamp((int)Math.Round(center), 0, inSize - 1);
                    from = nearest;
                    taps = [1];
                    total = 1;
                }

                start[o] = from;
                count[o] = taps.Length;
                offset[o] = values.Count;
                foreach (double tap in taps)
                {
                    values.Add((float)(tap / total));
                }
            }

            return new Weights(start, count, offset, [.. values]);
        }
    }
}
