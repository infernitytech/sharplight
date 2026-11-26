using Sharplight.VideoEnhancement;
using Sharplight.Vision;

namespace Sharplight.Architectures.VideoEnhancement;

/// <summary>The resolved sizes, batch length and settings of one video-enhancement request (the batches themselves follow the frames as they are read).</summary>
internal sealed record EnhancementPlan
{
    /// <summary>Width of the input frames in pixels.</summary>
    public required int InputWidth { get; init; }

    /// <summary>Height of the input frames in pixels.</summary>
    public required int InputHeight { get; init; }

    /// <summary>Output width in pixels.</summary>
    public required int Width { get; init; }

    /// <summary>Output height in pixels.</summary>
    public required int Height { get; init; }

    /// <summary>The width the graphs run at (a multiple of the size multiple; the result is cropped back).</summary>
    public required int PaddedWidth { get; init; }

    /// <summary>The height the graphs run at.</summary>
    public required int PaddedHeight { get; init; }

    /// <summary>Input frames per batch (every batch but the last holds this many); <see cref="int.MaxValue"/> for the whole clip of unknown length.</summary>
    public required int FramesPerBatch { get; init; }

    /// <summary>The temporal compression: padded batches hold 1 + k · this many frames.</summary>
    public required int FrameMultiple { get; init; }

    /// <summary>The frames of the clip when known (exact for frames in memory, else the request's estimate).</summary>
    public required long? TotalFrames { get; init; }

    /// <summary>The colour correction.</summary>
    public required VideoColorCorrection ColorCorrection { get; init; }

    /// <summary>Seed of the noise.</summary>
    public required long Seed { get; init; }

    /// <summary>The playback rate of the result.</summary>
    public required double FramesPerSecond { get; init; }

    /// <summary>The longest tile side the autoencoder runs at, or <see langword="null"/> to choose from the device's memory.</summary>
    public required int? TileSize { get; init; }

    /// <summary>The batches the clip is expected to have (exact when <see cref="TotalFrames"/> is), or 0 when unknown.</summary>
    public int ExpectedBatches => TotalFrames is { } total ? (int)Math.Max(1, (total + FramesPerBatch - 1) / FramesPerBatch) : 0;

    /// <summary>
    /// Plans a request from its first frame: the output size from the options (both sides, one side with the input's aspect ratio, or the
    /// scale), the padded size the graphs take, and batches of 1 + k · the frame multiple frames (the last one shorter and padded); a
    /// whole-clip request (frames per batch 0) is one batch up to the model's longest batch.
    /// </summary>
    /// <param name="first">The request's first frame (all others must have its size).</param>
    /// <param name="totalFrames">The frames of the clip when known.</param>
    /// <param name="framesPerSecond">The request's rate, or <see langword="null"/>.</param>
    /// <param name="options">The options.</param>
    /// <param name="defaults">The model's defaults.</param>
    /// <exception cref="ArgumentException">The frame is not 8-bit RGB, or an option is out of range.</exception>
    public static EnhancementPlan Create(ImageInput first, long? totalFrames, double? framesPerSecond, VideoEnhancementOptions options, VideoEnhancementDefaults defaults)
    {
        CheckFrame(first, first.Width, first.Height, 0);
        var (width, height) = OutputSize(first.Width, first.Height, options, defaults.Scale);
        int multiple = defaults.SizeMultiple;
        int frameMultiple = defaults.FrameMultiple;
        int perBatch = options.FramesPerBatch ?? defaults.FramesPerBatch;
        ArgumentOutOfRangeException.ThrowIfNegative(perBatch, nameof(options));
        if (options.TileSize is { } tile)
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(tile, multiple, nameof(options));
        }

        long? total = totalFrames is > 0 ? totalFrames : null;
        int longest = defaults.MaxFramesPerBatch > 0 ? defaults.MaxFramesPerBatch : int.MaxValue;
        if (perBatch == 0)
        {
            // The whole clip, unless it is longer than the model's longest batch (or of unknown length with such a limit).
            perBatch = total is { } count && count <= longest ? (int)count : longest;
        }

        if (total is not { } known || perBatch < known)
        {
            perBatch = perBatch == int.MaxValue ? perBatch : Math.Max(1, (perBatch - 1) / frameMultiple * frameMultiple + 1);
        }

        return new EnhancementPlan
        {
            InputWidth = first.Width,
            InputHeight = first.Height,
            Width = width,
            Height = height,
            PaddedWidth = RoundUp(width, multiple),
            PaddedHeight = RoundUp(height, multiple),
            FramesPerBatch = perBatch,
            FrameMultiple = frameMultiple,
            TotalFrames = total,
            ColorCorrection = options.ColorCorrection ?? defaults.ColorCorrection,
            Seed = options.Seed ?? Random.Shared.NextInt64(0, 1L << 53),
            FramesPerSecond = framesPerSecond is > 0 and var rate ? rate : 24,
            TileSize = options.TileSize,
        };
    }

    /// <summary>The batch of <paramref name="count"/> frames starting at input frame <paramref name="start"/>.</summary>
    /// <param name="index">The batch's index.</param>
    /// <param name="start">Its first input frame.</param>
    /// <param name="count">Its input frames.</param>
    public EnhancementBatch Batch(int index, long start, int count)
    {
        // One frame stays one (an image); otherwise pad to 1 + k·r frames, at least 1 + r (the official cut_videos).
        int padded = count == 1 ? 1 : Math.Max(FrameMultiple + 1, (count - 2) / FrameMultiple * FrameMultiple + FrameMultiple + 1);
        return new EnhancementBatch(index, start, count, padded, (padded - 1) / FrameMultiple + 1);
    }

    /// <summary>Checks that a frame of the request is 8-bit RGB of the clip's size.</summary>
    /// <param name="frame">The frame.</param>
    /// <param name="width">The clip's width.</param>
    /// <param name="height">The clip's height.</param>
    /// <param name="index">The frame's index (for the message).</param>
    /// <exception cref="ArgumentException">It is not.</exception>
    public static void CheckFrame(ImageInput frame, int width, int height, long index)
    {
        ArgumentNullException.ThrowIfNull(frame);
        if (frame.Width != width || frame.Height != height || frame.Rgb.Length != 3 * frame.Width * frame.Height)
        {
            throw new ArgumentException($"All frames must be {width}x{height} 8-bit RGB (frame {index} is {frame.Width}x{frame.Height}).");
        }
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
