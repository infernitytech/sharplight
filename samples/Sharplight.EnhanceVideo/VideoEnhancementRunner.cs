using System.Diagnostics;
using System.Globalization;
using Microsoft.Extensions.DependencyInjection;
using Sharplight.Imaging;
using Sharplight.Runtime;
using Sharplight.Samples;
using Sharplight.Tasks;
using Sharplight.Video;
using Sharplight.VideoEnhancement;

namespace Sharplight.EnhanceVideo;

/// <summary>Reads the input, loads the model through the <see cref="SampleRuntime"/> container, runs its <see cref="IVideoEnhancer"/> and writes the result.</summary>
internal sealed class VideoEnhancementRunner(EnhanceVideoSettings settings)
{
    public async Task<int> RunAsync(CancellationToken cancellationToken)
    {
        using var runtime = new SampleRuntime();
        var request = await ReadInputAsync(runtime.Services.GetRequiredService<VideoDecoderCatalog>(), cancellationToken);
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
        var result = await EnhanceAsync(enhancer, request, cancellationToken);
        await WriteAsync(runtime.Services.GetRequiredService<VideoEncoderCatalog>(), result, cancellationToken);
        return ExitCodes.Success;
    }

    /// <summary>A PNG becomes a one-frame request; anything else is decoded as a video.</summary>
    private async Task<VideoEnhancementRequest> ReadInputAsync(VideoDecoderCatalog decoders, CancellationToken cancellationToken)
    {
        if (!File.Exists(settings.Input))
        {
            throw new SampleException($"Input '{settings.Input}' does not exist.", ExitCodes.InvalidInput);
        }

        if (string.Equals(Path.GetExtension(settings.Input), ".png", StringComparison.OrdinalIgnoreCase))
        {
            return VideoEnhancementRequest.FromImage(PngCodec.Read(settings.Input));
        }

        if (!decoders.IsAvailable)
        {
            throw new SampleException("No video decoder is available (the bundled FFmpeg libraries are missing).", ExitCodes.InvalidInput);
        }

        var clip = await decoders.ReadClipAsync(settings.Input, new VideoDecodeOptions { Start = TimeSpan.FromSeconds(settings.Start), MaxFrames = settings.MaxFrames },
            cancellationToken);
        Console.WriteLine($"Read {clip.Frames.Count} frames of {clip.Width}x{clip.Height} at {clip.FramesPerSecond:F2} fps from {settings.Input}");
        return VideoEnhancementRequest.FromClip(clip);
    }

    /// <summary>One request with progress and timings per phase.</summary>
    private async Task<VideoEnhancementResult> EnhanceAsync(IVideoEnhancer enhancer, VideoEnhancementRequest request, CancellationToken cancellationToken)
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
        var phaseStart = new Dictionary<VideoEnhancementPhase, double>();
        VideoEnhancementResult? result = null;
        await foreach (var update in enhancer.EnhanceStreamingAsync(request, options, cancellationToken))
        {
            phaseStart.TryAdd(update.Phase, stopwatch.Elapsed.TotalSeconds);
            result = update.Result ?? result;
            Console.Write($"\r{update.Phase,-10} batch {update.Batch + 1}/{update.TotalBatches}  {stopwatch.Elapsed.TotalSeconds:F1} s   ");
        }

        Console.WriteLine();
        double encode = phaseStart.GetValueOrDefault(VideoEnhancementPhase.Restoring) - phaseStart.GetValueOrDefault(VideoEnhancementPhase.Encoding);
        double restore = phaseStart.GetValueOrDefault(VideoEnhancementPhase.Decoding) - phaseStart.GetValueOrDefault(VideoEnhancementPhase.Restoring);
        double decode = phaseStart.GetValueOrDefault(VideoEnhancementPhase.Completed) - phaseStart.GetValueOrDefault(VideoEnhancementPhase.Decoding);
        Console.WriteLine($"total {stopwatch.Elapsed.TotalSeconds:F1} s (encoding {encode:F1} s, restoring {restore:F1} s, decoding {decode:F1} s; first runs include compilation)");
        return result ?? throw new InvalidOperationException("Video enhancement produced no result.");
    }

    /// <summary>Writes a PNG (one frame), a video file (the encoder chosen by the extension) or, without an extension, numbered PNGs into a folder.</summary>
    private async Task WriteAsync(VideoEncoderCatalog encoders, VideoEnhancementResult result, CancellationToken cancellationToken)
    {
        string output = settings.Output;
        string extension = Path.GetExtension(output);
        if (result.Frames.Count == 1 && string.Equals(extension, ".png", StringComparison.OrdinalIgnoreCase))
        {
            PngCodec.Write(output, result.Frames[0]);
            Console.WriteLine($"wrote {output} ({result.Width}x{result.Height})");
            return;
        }

        if (extension.Length > 0)
        {
            if (encoders.FindForPath(output) is null)
            {
                throw new SampleException($"No available encoder writes '{extension}' files.", ExitCodes.InvalidInput);
            }

            await encoders.SaveAsync(result.ToClip(), output, cancellationToken: cancellationToken);
            Console.WriteLine($"wrote {output} ({result.Frames.Count} frames of {result.Width}x{result.Height} at {result.FramesPerSecond:F2} fps, seed {result.Seed})");
            return;
        }

        Directory.CreateDirectory(output);
        for (int i = 0; i < result.Frames.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            PngCodec.Write(Path.Combine(output, string.Create(CultureInfo.InvariantCulture, $"frame-{i:D4}.png")), result.Frames[i]);
        }

        Console.WriteLine($"wrote {result.Frames.Count} frames of {result.Width}x{result.Height} to {output}");
    }
}
