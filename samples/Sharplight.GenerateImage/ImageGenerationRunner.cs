using System.Diagnostics;
using Microsoft.Extensions.DependencyInjection;
using Sharplight.ImageGeneration;
using Sharplight.Imaging;
using Sharplight.Runtime;
using Sharplight.Samples;
using Sharplight.Tasks;

namespace Sharplight.GenerateImage;

/// <summary>Resolves an <see cref="IImageGenerator"/> from the service container and runs one request, writing PNGs.</summary>
internal sealed class ImageGenerationRunner(GenerateImageSettings settings)
{
    public async Task<int> RunAsync(CancellationToken cancellationToken)
    {
        var services = new ServiceCollection().AddSharplight();
        var provider = SampleRuntime.SelectProvider(settings.Cpu);
        var device = provider.OpenDevice(SampleRuntime.GetDevice(provider, settings.Device));
        services.AddImageGenerator(o =>
        {
            o.ModelPath = settings.Model;
            o.Devices = [new DeviceSlot { Device = device }];
            o.Log = message => Console.Error.WriteLine(message.Split('\n')[0]);
        });
        await using var container = services.BuildServiceProvider();
        var stopwatch = Stopwatch.StartNew();
        var generator = container.GetRequiredService<IImageGenerator>();
        Console.WriteLine($"Loaded {generator.Metadata.ModelName} ({generator.Metadata.Architecture}) on {device.Info.Name} in {stopwatch.Elapsed.TotalSeconds:F1} s");
        stopwatch.Restart();
        ImageGenerationResult? result = null;
        double firstStep = 0;
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

            Console.Write($"\rstep {update.Step}/{update.TotalSteps}  {stopwatch.Elapsed.TotalSeconds:F1} s");
            if (update.Preview is { } preview)
            {
                PngCodec.Write(InsertSuffix(settings.Output, $"preview-{update.Step:D3}"), preview);
            }
        }

        Console.WriteLine();
        var total = stopwatch.Elapsed.TotalSeconds;
        for (int i = 0; i < result!.Images.Count; i++)
        {
            string path = result.Images.Count == 1 ? settings.Output : InsertSuffix(settings.Output, i.ToString(System.Globalization.CultureInfo.InvariantCulture));
            PngCodec.Write(path, result.Images[i]);
            Console.WriteLine($"wrote {path} (seed {result.Seed + i})");
        }

        Console.WriteLine($"generation {total:F2} s (first step incl. compilation {firstStep:F2} s)");
        return ExitCodes.Success;
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
