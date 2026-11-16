using System.Diagnostics;
using Microsoft.Extensions.DependencyInjection;
using Sharplight.ImageGeneration;
using Sharplight.Imaging;
using Sharplight.Runtime;
using Sharplight.Samples;
using Sharplight.Tasks;
using Sharplight.Vision;

namespace Sharplight.GenerateImage;

/// <summary>Loads the model through the <see cref="SampleRuntime"/> container, creates its <see cref="IImageGenerator"/> and runs one request, writing PNGs.</summary>
internal sealed class ImageGenerationRunner(GenerateImageSettings settings)
{
    /// <summary>The companion files given on the command line, by auxiliary role (the rest are looked for next to the model).</summary>
    private Dictionary<string, string> Companions()
    {
        var companions = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var (role, path) in new[] { (AuxiliaryRoles.TextEncoder, settings.TextEncoder), (AuxiliaryRoles.SecondTextEncoder, settings.SecondTextEncoder),
            (AuxiliaryRoles.ThirdTextEncoder, settings.ThirdTextEncoder), (AuxiliaryRoles.Vae, settings.Vae) })
        {
            if (path is not null)
            {
                companions[role] = path;
            }
        }

        return companions;
    }

    public async Task<int> RunAsync(CancellationToken cancellationToken)
    {
        using var runtime = new SampleRuntime();
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
        using var generator = runtime.Services.GetRequiredService<TaskClientProvider>().Create<IImageGenerator>(model);
        Console.WriteLine($"Loaded {generator.Metadata.ModelName} ({generator.Metadata.Architecture}) on {device.Info.Name} in {stopwatch.Elapsed.TotalSeconds:F1} s");
        ImageGenerationResult? result = null;
        for (int run = 0; run < Math.Max(1, settings.Runs); run++)
        {
            result = await RunOnceAsync(generator, run, cancellationToken);
        }

        for (int i = 0; i < result!.Images.Count; i++)
        {
            string path = result.Images.Count == 1 ? settings.Output : InsertSuffix(settings.Output, i.ToString(System.Globalization.CultureInfo.InvariantCulture));
            PngCodec.Write(path, result.Images[i]);
            Console.WriteLine($"wrote {path} (seed {result.Seed + i})");
        }

        return ExitCodes.Success;
    }

    /// <summary>One generation, timed: the first step includes encoding the prompts (and compiling on the first run), the last update the decode.</summary>
    private async Task<ImageGenerationResult> RunOnceAsync(IImageGenerator generator, int run, CancellationToken cancellationToken)
    {
        var stopwatch = Stopwatch.StartNew();
        ImageGenerationResult? result = null;
        double firstStep = 0;
        double lastStep = 0;
        int steps = 0;
        await foreach (var update in generator.GenerateStreamingAsync(Request(), Options(), cancellationToken))
        {
            if (update.Result is { } finished)
            {
                result = finished;
                continue;
            }

            if (update.Step == 1)
            {
                firstStep = stopwatch.Elapsed.TotalSeconds;
            }

            lastStep = stopwatch.Elapsed.TotalSeconds;
            steps = update.Step;
            Console.Write($"\rstep {update.Step}/{update.TotalSteps}  {stopwatch.Elapsed.TotalSeconds:F1} s");
            if (update.Preview is { } preview)
            {
                PngCodec.Write(InsertSuffix(settings.Output, $"preview-{update.Step:D3}"), preview);
            }
        }

        double total = stopwatch.Elapsed.TotalSeconds;
        double perStep = steps > 1 ? (lastStep - firstStep) / (steps - 1) : 0;
        Console.WriteLine();
        Console.WriteLine($"run {run + 1}: {total:F2} s (prompt encoding + first step {firstStep:F2} s{(run == 0 ? ", incl. compilation" : "")}, "
            + $"{perStep:F2} s per further step, decode {total - lastStep:F2} s)");
        return result ?? throw new InvalidOperationException("Image generation produced no result.");
    }

    private ImageGenerationRequest Request() => new()
    {
        Prompt = settings.Prompt,
        NegativePrompt = settings.NegativePrompt,
        Image = settings.InitImage is null ? null : ReadImage(settings.InitImage),
        Mask = settings.Mask is null ? null : ReadImage(settings.Mask),
        Strength = settings.Strength,
    };

    private ImageGenerationOptions Options() => new()
    {
        Width = settings.Width,
        Height = settings.Height,
        Steps = settings.Steps,
        GuidanceScale = settings.Guidance,
        Seed = settings.Seed,
        Count = settings.Count,
        Scheduler = settings.Sampler,
        InitialNoise = settings.Noise is null ? null : ReadFloats(settings.Noise),
        PreviewInterval = settings.PreviewInterval,
    };

    private static Vision.ImageInput ReadImage(string path)
    {
        if (!File.Exists(path))
        {
            throw new SampleException($"Image '{path}' does not exist.", ExitCodes.InvalidInput);
        }

        return PngCodec.Read(path);
    }

    private static float[] ReadFloats(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var values = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, values, 0, values.Length * sizeof(float));
        return values;
    }

    private static string InsertSuffix(string path, string suffix)
        => Path.Combine(Path.GetDirectoryName(path) ?? "", $"{Path.GetFileNameWithoutExtension(path)}-{suffix}{Path.GetExtension(path)}");
}
