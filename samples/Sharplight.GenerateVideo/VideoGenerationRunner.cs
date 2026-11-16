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
    /// <summary>Formats tried, in order, when the requested one has no encoder on this machine.</summary>
    private static readonly string[] s_fallbacks = [".webm", ".mp4", ".apng", ".gif"];

    public async Task<int> RunAsync(CancellationToken cancellationToken)
    {
        using var runtime = new SampleRuntime();
        var encoders = runtime.Services.GetRequiredService<VideoEncoderCatalog>();
        string output = ResolveOutput(encoders);
        var provider = runtime.SelectProvider(settings.Cpu);
        using var device = provider.OpenDevice(SampleRuntime.GetDevice(provider, settings.Device));
        var stopwatch = Stopwatch.StartNew();
        using var model = await TaskModel.LoadAsync(runtime.Loader, new TaskLoadOptions
        {
            ModelPath = settings.Model,
            Auxiliary = Companions(),
            AllowReducedPrecision = settings.ReducedPrecision,
            Devices = [new DeviceSlot { Device = device }],
            Log = message => Console.Error.WriteLine(message.Split('\n')[0]),
        }, cancellationToken);
        using var generator = runtime.Services.GetRequiredService<TaskClientProvider>().Create<IVideoGenerator>(model);
        Console.WriteLine($"Loaded {generator.Metadata.ModelName} ({generator.Metadata.Architecture}) on {device.Info.Name} in {stopwatch.Elapsed.TotalSeconds:F1} s");
        var result = await GenerateAsync(generator, cancellationToken);
        await WriteAsync(encoders, output, result, cancellationToken);
        return ExitCodes.Success;
    }

    /// <summary>The auxiliary sources: a companion folder for the text encoder and VAE, a second (low-noise) expert file, an image encoder folder.</summary>
    private Dictionary<string, string> Companions()
    {
        var companions = new Dictionary<string, string>(StringComparer.Ordinal);
        if (settings.Companions is { } folder)
        {
            companions[AuxiliaryRoles.TextEncoder] = folder;
            companions[AuxiliaryRoles.Vae] = folder;
        }

        if (settings.SecondTransformer is { } second)
        {
            companions[AuxiliaryRoles.SecondTransformer] = second;
        }

        if (settings.ImageEncoder is { } imageEncoder)
        {
            companions[AuxiliaryRoles.ImageEncoder] = imageEncoder;
        }

        return companions;
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
        LastImage = settings.LastImage is null ? null : ReadImage(settings.LastImage),
    };

    private VideoGenerationOptions Options() => new()
    {
        Width = settings.Width,
        Height = settings.Height,
        Frames = settings.Frames,
        FramesPerSecond = settings.FramesPerSecond,
        Steps = settings.Steps,
        GuidanceScale = settings.Guidance,
        GuidanceScale2 = settings.Guidance2,
        Seed = settings.Seed,
        Scheduler = settings.Sampler,
        InitialNoise = settings.Noise is null ? null : ReadFloats(settings.Noise),
    };

    /// <summary>
    /// The output path, checked before the model loads: a known video format whose encoder is missing on this machine (WebM without the
    /// native libsharplight_video library, MP4 without a system ffmpeg) falls back to another available format with a message; an unknown
    /// extension is an error.
    /// </summary>
    private string ResolveOutput(VideoEncoderCatalog encoders)
    {
        string output = settings.Output;
        if (!Path.HasExtension(output) || encoders.FindForPath(output) is not null)
        {
            return output;
        }

        string extension = Path.GetExtension(output).ToLowerInvariant();
        string available = string.Join(", ", encoders.Available.SelectMany(e => e.FileExtensions));
        string? reason = extension switch
        {
            ".webm" => "WebM needs Sharplight's native video library (libsharplight_video), which this build does not include (build it with native/sharplight_video/build.sh)",
            ".mp4" or ".m4v" => "MP4 needs an ffmpeg executable on the PATH",
            _ => null,
        };
        string? fallback = s_fallbacks.FirstOrDefault(candidate => encoders.FindForPath("video" + candidate) is not null);
        if (reason is null || fallback is null)
        {
            throw new SampleException($"No available encoder writes '{extension}' files (available: {available}).", ExitCodes.InvalidInput);
        }

        string replaced = Path.ChangeExtension(output, fallback);
        Console.Error.WriteLine($"warning: {reason}; writing {replaced} instead.");
        return replaced;
    }

    /// <summary>Writes a video file (the encoder chosen by the extension) or, without an extension, numbered PNGs into a folder.</summary>
    private async Task WriteAsync(VideoEncoderCatalog encoders, string output, VideoGenerationResult result, CancellationToken cancellationToken)
    {
        if (Path.HasExtension(output))
        {
            await encoders.SaveAsync(result.ToClip(), output, cancellationToken: cancellationToken);
            Console.WriteLine($"wrote {output} ({result.Frames.Count} frames, {result.Duration.TotalSeconds:F2} s at {result.FramesPerSecond} fps, seed {result.Seed})");
            return;
        }

        Directory.CreateDirectory(output);
        for (int i = 0; i < result.Frames.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            PngCodec.Write(Path.Combine(output, string.Create(CultureInfo.InvariantCulture, $"frame-{i:D4}.png")), result.Frames[i]);
        }

        Console.WriteLine($"wrote {result.Frames.Count} frames ({result.Duration.TotalSeconds:F2} s at {result.FramesPerSecond} fps, seed {result.Seed}) to {output}");
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
