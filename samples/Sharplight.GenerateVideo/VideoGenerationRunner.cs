using System.Diagnostics;
using System.Globalization;
using Microsoft.Extensions.DependencyInjection;
using Sharplight.Imaging;
using Sharplight.Runtime;
using Sharplight.Samples;
using Sharplight.Tasks;
using Sharplight.Video;
using Sharplight.Vision;
using Sharplight.VideoGeneration;

namespace Sharplight.GenerateVideo;

/// <summary>Loads the model through the <see cref="SampleRuntime"/> container, creates its <see cref="IVideoGenerator"/> and writes the video.</summary>
internal sealed class VideoGenerationRunner(GenerateVideoSettings settings)
{
    public async Task<int> RunAsync(CancellationToken cancellationToken)
    {
        using var runtime = new SampleRuntime();
        var provider = runtime.SelectProvider(settings.Cpu);
        using var device = provider.OpenDevice(SampleRuntime.GetDevice(provider, settings.Device));
        var stopwatch = Stopwatch.StartNew();
        using var model = await TaskModel.LoadAsync(runtime.Loader, new TaskLoadOptions
        {
            ModelPath = settings.Model,
            Auxiliary = settings.Companions is { } folder
                ? new Dictionary<string, string> { [AuxiliaryRoles.TextEncoder] = folder, [AuxiliaryRoles.Vae] = folder }
                : new Dictionary<string, string>(),
            AllowReducedPrecision = settings.ReducedPrecision,
            Devices = [new DeviceSlot { Device = device }],
            Log = message => Console.Error.WriteLine(message.Split('\n')[0]),
        }, cancellationToken);
        using var generator = runtime.Services.GetRequiredService<TaskClientProvider>().Create<IVideoGenerator>(model);
        Console.WriteLine($"Loaded {generator.Metadata.ModelName} ({generator.Metadata.Architecture}) on {device.Info.Name} in {stopwatch.Elapsed.TotalSeconds:F1} s");
        var result = await GenerateAsync(generator, cancellationToken);
        await WriteAsync(runtime.Services.GetRequiredService<VideoEncoderCatalog>(), result, cancellationToken);
        return ExitCodes.Success;
    }

    /// <summary>One generation with progress and timings per phase.</summary>
    private async Task<VideoGenerationResult> GenerateAsync(IVideoGenerator generator, CancellationToken cancellationToken)
    {
        var stopwatch = Stopwatch.StartNew();
        var phaseStart = new Dictionary<VideoGenerationPhase, double>();
        VideoGenerationResult? result = null;
        await foreach (var update in generator.GenerateStreamingAsync(Request(), Options(), cancellationToken))
        {
            phaseStart.TryAdd(update.Phase, stopwatch.Elapsed.TotalSeconds);
            result = update.Result ?? result;
            Console.Write($"\r{update.Phase,-12} {update.Step}/{update.TotalSteps}  {stopwatch.Elapsed.TotalSeconds:F1} s   ");
        }

        Console.WriteLine();
        double denoise = phaseStart.GetValueOrDefault(VideoGenerationPhase.Decoding) - phaseStart.GetValueOrDefault(VideoGenerationPhase.Denoising);
        double decode = phaseStart.GetValueOrDefault(VideoGenerationPhase.Completed) - phaseStart.GetValueOrDefault(VideoGenerationPhase.Decoding);
        Console.WriteLine($"total {stopwatch.Elapsed.TotalSeconds:F1} s (conditioning {phaseStart.GetValueOrDefault(VideoGenerationPhase.Denoising):F1} s, "
            + $"denoising {denoise:F1} s, decoding {decode:F1} s)");
        return result ?? throw new InvalidOperationException("Video generation produced no result.");
    }

    private VideoGenerationRequest Request() => new()
    {
        Prompt = settings.Prompt,
        NegativePrompt = settings.NegativePrompt,
        Image = settings.Image is null ? null : ReadImage(settings.Image),
    };

    private VideoGenerationOptions Options() => new()
    {
        Width = settings.Width,
        Height = settings.Height,
        Frames = settings.Frames,
        FramesPerSecond = settings.FramesPerSecond,
        Steps = settings.Steps,
        GuidanceScale = settings.Guidance,
        Seed = settings.Seed,
        Scheduler = settings.Sampler,
        InitialNoise = settings.Noise is null ? null : ReadFloats(settings.Noise),
    };

    /// <summary>Writes a video file (the encoder chosen by the extension) or, without an extension, numbered PNGs into a folder.</summary>
    private async Task WriteAsync(VideoEncoderCatalog encoders, VideoGenerationResult result, CancellationToken cancellationToken)
    {
        if (Path.HasExtension(settings.Output))
        {
            if (encoders.FindForPath(settings.Output) is not { IsAvailable: true })
            {
                string available = string.Join(", ", encoders.Available.SelectMany(e => e.FileExtensions));
                throw new SampleException($"No available encoder writes '{Path.GetExtension(settings.Output)}' files (available: {available}).", ExitCodes.InvalidInput);
            }

            await encoders.SaveAsync(result.ToClip(), settings.Output, cancellationToken: cancellationToken);
            Console.WriteLine($"wrote {settings.Output} ({result.Frames.Count} frames, {result.Duration.TotalSeconds:F2} s at {result.FramesPerSecond} fps, seed {result.Seed})");
            return;
        }

        Directory.CreateDirectory(settings.Output);
        for (int i = 0; i < result.Frames.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            PngCodec.Write(Path.Combine(settings.Output, string.Create(CultureInfo.InvariantCulture, $"frame-{i:D4}.png")), result.Frames[i]);
        }

        Console.WriteLine($"wrote {result.Frames.Count} frames ({result.Duration.TotalSeconds:F2} s at {result.FramesPerSecond} fps, seed {result.Seed}) to {settings.Output}");
    }

    private static ImageInput ReadImage(string path)
        => File.Exists(path) ? PngCodec.Read(path) : throw new SampleException($"Image '{path}' does not exist.", ExitCodes.InvalidInput);

    private static float[] ReadFloats(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var values = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, values, 0, values.Length * sizeof(float));
        return values;
    }
}
