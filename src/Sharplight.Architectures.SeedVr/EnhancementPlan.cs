using Sharplight.VideoEnhancement;

namespace Sharplight.Architectures.SeedVr;

/// <summary>The resolved sizes, batches and settings of one video-enhancement request.</summary>
internal sealed record EnhancementPlan
{
    /// <summary>Output width in pixels.</summary>
    public required int Width { get; init; }

    /// <summary>Output height in pixels.</summary>
    public required int Height { get; init; }

    /// <summary>The width the graphs run at (a multiple of the size multiple; the result is cropped back).</summary>
    public required int PaddedWidth { get; init; }

    /// <summary>The height the graphs run at.</summary>
    public required int PaddedHeight { get; init; }

    /// <summary>The batches, in order.</summary>
    public required IReadOnlyList<EnhancementBatch> Batches { get; init; }

    /// <summary>The colour correction.</summary>
    public required VideoColorCorrection ColorCorrection { get; init; }

    /// <summary>Seed of the noise.</summary>
    public required long Seed { get; init; }

    /// <summary>The playback rate of the result.</summary>
    public required double FramesPerSecond { get; init; }

    /// <summary>
    /// Plans a request: the output size from the options (both sides, one side with the input's aspect ratio, or the scale), the padded size
    /// the graphs take, and batches of 1 + k · the frame multiple frames (the last one shorter and padded).
    /// </summary>
    /// <param name="request">The request.</param>
    /// <param name="options">Its options.</param>
    /// <param name="defaults">The model's defaults.</param>
    /// <exception cref="ArgumentException">The frames are missing or differ in size, or an option is out of range.</exception>
    public static EnhancementPlan Create(VideoEnhancementRequest request, VideoEnhancementOptions options, VideoEnhancementDefaults defaults)
    {
        ArgumentNullException.ThrowIfNull(request.Frames);
        if (request.Frames.Count == 0)
        {
            throw new ArgumentException("A video-enhancement request needs at least one frame.", nameof(request));
        }

        var first = request.Frames[0];
        if (request.Frames.Any(f => f.Width != first.Width || f.Height != first.Height || f.Rgb.Length != 3 * f.Width * f.Height))
        {
            throw new ArgumentException($"All frames must be {first.Width}x{first.Height} 8-bit RGB.", nameof(request));
        }

        var (width, height) = OutputSize(first.Width, first.Height, options, defaults.Scale);
        int multiple = defaults.SizeMultiple;
        int frameMultiple = defaults.FrameMultiple;
        int total = request.Frames.Count;
        int perBatch = options.FramesPerBatch ?? defaults.FramesPerBatch;
        ArgumentOutOfRangeException.ThrowIfNegative(perBatch, nameof(options));
        perBatch = perBatch == 0 || perBatch >= total ? total : Math.Max(1, (perBatch - 1) / frameMultiple * frameMultiple + 1);
        var batches = new List<EnhancementBatch>();
        for (int start = 0; start < total; start += perBatch)
        {
            int count = Math.Min(perBatch, total - start);
            // One frame stays one (an image); otherwise pad to 1 + k·r frames, at least 1 + r (the official cut_videos).
            int padded = count == 1 ? 1 : Math.Max(frameMultiple + 1, (count - 2) / frameMultiple * frameMultiple + frameMultiple + 1);
            batches.Add(new EnhancementBatch(start, count, padded, (padded - 1) / frameMultiple + 1));
        }

        return new EnhancementPlan
        {
            Width = width,
            Height = height,
            PaddedWidth = RoundUp(width, multiple),
            PaddedHeight = RoundUp(height, multiple),
            Batches = batches,
            ColorCorrection = options.ColorCorrection ?? defaults.ColorCorrection,
            Seed = options.Seed ?? Random.Shared.NextInt64(0, 1L << 53),
            FramesPerSecond = request.FramesPerSecond is > 0 and var rate ? rate : 24,
        };
    }

    private static (int Width, int Height) OutputSize(int inputWidth, int inputHeight, VideoEnhancementOptions options, float defaultScale)
    {
        if (options.Width is { } w && options.Height is { } h)
        {
            return (Positive(w), Positive(h));
        }

        if (options.Width is { } onlyWidth)
        {
            return (Positive(onlyWidth), Math.Max(1, (int)Math.Round((double)onlyWidth * inputHeight / inputWidth)));
        }

        if (options.Height is { } onlyHeight)
        {
            return (Math.Max(1, (int)Math.Round((double)onlyHeight * inputWidth / inputHeight)), Positive(onlyHeight));
        }

        float scale = options.Scale ?? defaultScale;
        if (!(scale > 0) || !float.IsFinite(scale))
        {
            throw new ArgumentOutOfRangeException(nameof(options), scale, "The scale must be positive.");
        }

        return (Math.Max(1, (int)Math.Round(inputWidth * (double)scale)), Math.Max(1, (int)Math.Round(inputHeight * (double)scale)));
    }

    private static int Positive(int value) => value > 0 ? value : throw new ArgumentOutOfRangeException(nameof(value), value, "Sizes must be positive.");

    private static int RoundUp(int value, int multiple) => (value + multiple - 1) / multiple * multiple;
}
