using System.Diagnostics;
using Microsoft.Extensions.DependencyInjection;
using Sharplight.Imaging;
using Sharplight.Runtime;
using Sharplight.Samples;
using Sharplight.Tasks;
using Sharplight.Video;
using Sharplight.VideoEnhancement;

namespace Sharplight.EnhanceVideo;

/// <summary>
/// Loads the model through the <see cref="SampleRuntime"/> container and streams the input through its <see cref="IVideoEnhancer"/>: a video is
/// decoded frame by frame, enhanced batch by batch and encoded as the batches finish (with the input's sound track), so a clip of any length
/// runs in the memory of a few batches.
/// </summary>
internal sealed class VideoEnhancementRunner(EnhanceVideoSettings settings)
{
    public async Task<int> RunAsync(CancellationToken cancellationToken)
    {
        using var runtime = new SampleRuntime();
        var request = await OpenInputAsync(runtime.Services.GetRequiredService<VideoDecoderCatalog>(), cancellationToken);
        var provider = runtime.SelectProvider(settings.Cpu);
        using var device = provider.OpenDevice(SampleRuntime.GetDevice(provider, settings.Device));
        var stopwatch = Stopwatch.StartNew();
        using var model = await TaskModel.LoadAsync(runtime.Loader, new TaskLoadOptions
        {
            ModelPath = settings.Model,
            AllowReducedPrecision = settings.ReducedPrecision,
            Devices = [new DeviceSlot { Device = device }],
            Log = message => Console.Error.WriteLine(message.Split('\n')[0]),
        }, cancellationToken);
        using var enhancer = runtime.Services.GetRequiredService<TaskClientProvider>().Create<IVideoEnhancer>(model);
        Console.WriteLine($"Loaded {enhancer.Metadata.ModelName} ({enhancer.Metadata.Architecture}) on {device.Info.Name} in {stopwatch.Elapsed.TotalSeconds:F1} s");
        await using var output = new EnhancedFrameOutput(settings, runtime.Services.GetRequiredService<VideoEncoderCatalog>());
        await EnhanceAsync(enhancer, request, output, cancellationToken);
        return ExitCodes.Success;
    }

    /// <summary>A PNG becomes a one-frame request; anything else is a video whose frames are decoded while the enhancement runs.</summary>
    private async Task<VideoEnhancementRequest> OpenInputAsync(VideoDecoderCatalog decoders, CancellationToken cancellationToken)
    {
        if (!File.Exists(settings.Input))
        {
            throw new SampleException($"Input '{settings.Input}' does not exist.", ExitCodes.InvalidInput);
        }

        if (string.Equals(Path.GetExtension(settings.Input), ".png", StringComparison.OrdinalIgnoreCase))
        {
            return VideoEnhancementRequest.FromImage(PngCodec.Read(settings.Input));
        }

        if (!decoders.TryGet(out var decoder))
        {
            throw new SampleException("No video decoder is available (the bundled FFmpeg libraries are missing).", ExitCodes.InvalidInput);
        }

        var info = await decoder.ProbeAsync(settings.Input, cancellationToken);
        var range = new VideoDecodeOptions { Start = TimeSpan.FromSeconds(settings.Start), MaxFrames = settings.MaxFrames };
        long expected = Math.Max(0, info.FrameCount - (long)Math.Round(settings.Start * info.FramesPerSecond));
        expected = settings.MaxFrames is { } most ? Math.Min(expected, most) : expected;
        Console.WriteLine($"Reading {settings.Input}: {info.Width}x{info.Height} at {info.FramesPerSecond:F2} fps, ~{expected} frames");
        var frames = decoder.ReadFramesAsync(settings.Input, range, cancellationToken).Select(frame => frame.ToImage());
        return VideoEnhancementRequest.FromSource(frames, info.FramesPerSecond > 0 ? info.FramesPerSecond : null, expected > 0 ? expected : null);
    }

    /// <summary>One streamed request: progress per batch, the finished frames written as they arrive.</summary>
    private async Task EnhanceAsync(IVideoEnhancer enhancer, VideoEnhancementRequest request, EnhancedFrameOutput output, CancellationToken cancellationToken)
    {
        var options = new VideoEnhancementOptions
        {
            Width = settings.Width,
            Height = settings.Height,
            Scale = settings.Scale,
            FramesPerBatch = settings.FramesPerBatch,
            ColorCorrection = settings.ColorCorrection,
            Seed = settings.Seed,
        };
        var stopwatch = Stopwatch.StartNew();
        await foreach (var update in enhancer.EnhanceStreamingAsync(request, options, cancellationToken))
        {
            if (update.Frames is { } frames)
            {
                await output.WriteAsync(frames, update, cancellationToken);
            }

            string total = update.TotalFrames is { } count ? $"/{count}" : "";
            Console.Write($"\r{update.Phase,-10} batch {update.Batch + 1}  frames {update.FinishedFrames}{total}  {stopwatch.Elapsed.TotalSeconds:F1} s   ");
        }

        Console.WriteLine();
        await output.CompleteAsync(cancellationToken);
        Console.WriteLine($"total {stopwatch.Elapsed.TotalSeconds:F1} s (the first batches include compilation)");
    }
}
