using System.Runtime.CompilerServices;
using Sharplight.Vision;

namespace Sharplight.Architectures.SeedVr;

/// <summary>
/// Conversions between 8-bit frames and the planar clips the graphs take ([F, 3, H, W] floats in [−1, 1]): antialiased bicubic resizing
/// (PyTorch's <c>interpolate(mode="bicubic", antialias=True)</c>, torchvision's default for tensors) clamped to [0, 1], edge padding to the
/// network's size multiple, and the way back with rounding half to even.
/// </summary>
internal static class ClipPixels
{
    /// <summary>
    /// Resizes <paramref name="frames"/> to <paramref name="width"/> × <paramref name="height"/>, pads them by repeating the last row and column
    /// to <paramref name="paddedWidth"/> × <paramref name="paddedHeight"/> and the clip by repeating its last frame to <paramref name="paddedFrames"/>.
    /// </summary>
    /// <returns>The clip [paddedFrames, 3, paddedHeight, paddedWidth] in [−1, 1].</returns>
    public static float[] Prepare(IReadOnlyList<ImageInput> frames, int width, int height, int paddedWidth, int paddedHeight, int paddedFrames)
    {
        int planeSize = paddedWidth * paddedHeight;
        long frameSize = 3L * planeSize;
        var clip = new float[paddedFrames * frameSize];
        var source = frames[0];
        var (rowStart, rowCount, rowWeights, rowMax) = ImageResampling.Coefficients(source.Height, height);
        var (columnStart, columnCount, columnWeights, columnMax) = ImageResampling.Coefficients(source.Width, width);
        var rowTaps = new ResampleTaps(rowStart, rowCount, rowWeights, rowMax);
        var columnTaps = new ResampleTaps(columnStart, columnCount, columnWeights, columnMax);
        Parallel.For(0, frames.Count, f => ResizeFrame(frames[f], clip.AsSpan((int)(f * frameSize), (int)frameSize), width, height, paddedWidth, paddedHeight,
            rowTaps, columnTaps));
        for (int f = frames.Count; f < paddedFrames; f++)
        {
            clip.AsSpan((int)((frames.Count - 1) * frameSize), (int)frameSize).CopyTo(clip.AsSpan((int)(f * frameSize)));
        }

        return clip;
    }

    /// <summary>
    /// Frames [n, 3, paddedHeight, paddedWidth] in [0, 1] (or in [−1, 1] when <paramref name="signed"/>) → 8-bit frames, cropped to
    /// <paramref name="width"/> × <paramref name="height"/>, the first <paramref name="count"/> frames.
    /// </summary>
    public static List<ImageInput> ToImages(float[] clip, int count, int paddedWidth, int paddedHeight, int width, int height, bool signed)
    {
        long frameSize = 3L * paddedWidth * paddedHeight;
        var images = new ImageInput[count];
        Parallel.For(0, count, f =>
        {
            var rgb = new byte[3 * width * height];
            var frame = clip.AsSpan((int)(f * frameSize), (int)frameSize);
            for (int c = 0; c < 3; c++)
            {
                var plane = frame.Slice(c * paddedWidth * paddedHeight, paddedWidth * paddedHeight);
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        float v = plane[y * paddedWidth + x];
                        float unit = signed ? Math.Clamp(v, -1f, 1f) * 0.5f + 0.5f : Math.Clamp(v, 0f, 1f);
                        rgb[(y * width + x) * 3 + c] = (byte)MathF.Round(unit * 255f, MidpointRounding.ToEven);
                    }
                }
            }

            images[f] = new ImageInput(width, height, rgb);
        });
        return [.. images];
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void ResizeFrame(ImageInput frame, Span<float> output, int width, int height, int paddedWidth, int paddedHeight, ResampleTaps rows,
        ResampleTaps columns)
    {
        int inWidth = frame.Width;
        int inHeight = frame.Height;
        var horizontal = new float[inHeight * width];
        int plane = paddedWidth * paddedHeight;
        for (int c = 0; c < 3; c++)
        {
            for (int y = 0; y < inHeight; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    float sum = 0;
                    int start = columns.Start[x];
                    for (int i = 0; i < columns.Count[x]; i++)
                    {
                        sum += columns.Weights[x * columns.Max + i] * (frame.Rgb[(y * inWidth + start + i) * 3 + c] / 255f);
                    }

                    horizontal[y * width + x] = sum;
                }
            }

            var target = output.Slice(c * plane, plane);
            for (int y = 0; y < paddedHeight; y++)
            {
                int row = Math.Min(y, height - 1);
                int start = rows.Start[row];
                for (int x = 0; x < width; x++)
                {
                    float sum = 0;
                    for (int i = 0; i < rows.Count[row]; i++)
                    {
                        sum += rows.Weights[row * rows.Max + i] * horizontal[(start + i) * width + x];
                    }

                    target[y * paddedWidth + x] = Math.Clamp(sum, 0f, 1f) * 2f - 1f;
                }

                float edge = target[y * paddedWidth + width - 1];
                target.Slice(y * paddedWidth + width, paddedWidth - width).Fill(edge);
            }
        }
    }
}
