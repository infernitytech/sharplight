using Sharplight.Vision;

namespace Sharplight.Imaging;

/// <summary>
/// Applies an <see cref="ImageProcessorConfig"/> to images: antialiased resize (Pillow semantics: 8-bit passes, rounding and clamping like
/// Hugging Face's image processors and torchvision), center crop, rescale and normalization to planar floats. Thread-safe.
/// </summary>
/// <param name="config">The preprocessing configuration.</param>
public sealed class ImageProcessor(ImageProcessorConfig config)
{
    /// <summary>The key of the feature-extractor component's configuration that holds the <see cref="ImageProcessorConfig"/> as JSON.</summary>
    public const string ComponentKey = "imageProcessor";

    /// <summary>The configuration.</summary>
    public ImageProcessorConfig Config { get; } = config ?? throw new ArgumentNullException(nameof(config));

    /// <summary>The size of the model input for an image of the given size.</summary>
    /// <param name="width">Source width.</param>
    /// <param name="height">Source height.</param>
    public (int Width, int Height) OutputSize(int width, int height)
    {
        var (rw, rh) = ResizedSize(width, height);
        return (Config.CropWidth > 0 ? Config.CropWidth : rw, Config.CropHeight > 0 ? Config.CropHeight : rh);
    }

    /// <summary>Resizes, crops and normalizes an image.</summary>
    /// <param name="image">The image.</param>
    /// <exception cref="ArgumentException">The image is invalid.</exception>
    public ProcessedImage Process(ImageInput image)
    {
        ArgumentNullException.ThrowIfNull(image);
        image.Validate();
        var (rw, rh) = ResizedSize(image.Width, image.Height);
        byte[] rgb = rw == image.Width && rh == image.Height ? image.Rgb : Resize(image, rw, rh);
        int cw = Config.CropWidth > 0 ? Config.CropWidth : rw;
        int ch = Config.CropHeight > 0 ? Config.CropHeight : rh;
        // Hugging Face's center_crop offset is (size - crop) / 2 rounded toward zero, torchvision's the nearest integer (ties to even); images smaller than the crop are zero padded.
        int left = CropOffset(rw - cw);
        int top = CropOffset(rh - ch);
        var planes = new float[3L * cw * ch];
        int plane = cw * ch;
        for (int c = 0; c < 3; c++)
        {
            float mean = Config.Mean.Count == 3 ? Config.Mean[c] : 0f;
            float std = Config.Std.Count == 3 ? Config.Std[c] : 1f;
            float scale = Config.Rescale / std;
            float bias = -mean / std;
            float pad = Config.PadColor * scale + bias;
            for (int y = 0; y < ch; y++)
            {
                int sy = y + top;
                for (int x = 0; x < cw; x++)
                {
                    int sx = x + left;
                    bool inside = sy >= 0 && sy < rh && sx >= 0 && sx < rw;
                    planes[c * plane + y * cw + x] = inside ? rgb[(sy * rw + sx) * 3 + c] * scale + bias : pad;
                }
            }
        }

        return new ProcessedImage(planes, cw, ch, image.Width, image.Height, (float)rw / image.Width, (float)rh / image.Height, left, top);
    }

    private int CropOffset(int excess) => Config.CropOffsetNearest ? (int)Math.Round(excess / 2.0, MidpointRounding.ToEven) : excess / 2;

    private (int Width, int Height) ResizedSize(int width, int height)
    {
        switch (Config.ResizeMode)
        {
            case ImageResizeMode.Exact:
                return (Config.Width, Config.Height);
            case ImageResizeMode.ShortestEdge:
            {
                int shortest = Math.Min(width, height);
                return width <= height
                    ? (Config.ShortestEdge, (int)(Config.ShortestEdge * (long)height / shortest))
                    : ((int)(Config.ShortestEdge * (long)width / shortest), Config.ShortestEdge);
            }

            case ImageResizeMode.LongestEdge:
            {
                int longest = Math.Max(width, height);
                return width >= height
                    ? (Config.LongestEdge, Math.Max(1, (int)(Config.LongestEdge * (long)height / longest)))
                    : (Math.Max(1, (int)(Config.LongestEdge * (long)width / longest)), Config.LongestEdge);
            }

            case ImageResizeMode.Contain:
            {
                double ratio = Math.Min((double)Config.Width / width, (double)Config.Height / height);
                return (Math.Max(1, (int)Math.Round(width * ratio, MidpointRounding.ToEven)), Math.Max(1, (int)Math.Round(height * ratio, MidpointRounding.ToEven)));
            }

            default:
                return (width, height);
        }
    }

    private byte[] Resize(ImageInput image, int dw, int dh)
    {
        if (Config.Filter != ImageResampleFilter.Nearest)
        {
            return VisionPreprocessor.Resize(image.Rgb, image.Width, image.Height, dw, dh, Config.Filter);
        }

        var dst = new byte[3L * dw * dh];
        for (int y = 0; y < dh; y++)
        {
            int sy = Math.Min((int)((y + 0.5) * image.Height / dh), image.Height - 1);
            for (int x = 0; x < dw; x++)
            {
                int sx = Math.Min((int)((x + 0.5) * image.Width / dw), image.Width - 1);
                Array.Copy(image.Rgb, (sy * image.Width + sx) * 3, dst, (y * dw + x) * 3, 3);
            }
        }

        return dst;
    }
}
