using Microsoft.Extensions.DependencyInjection;
using Sharplight.Imaging;
using Sharplight.Runtime;
using Sharplight.Samples;
using Sharplight.Tasks;

namespace Sharplight.Vision.Sample;

/// <summary>
/// Runs <c>sharplight-vision</c>: registers Sharplight and one task client for the model with the service container (the model loads on first
/// use), reads the PNG, runs the task and prints the result; detections are also drawn into the output image.
/// </summary>
public static class VisionApplication
{
    /// <summary>Classifies the image, zero-shot when <c>--label</c> prompts are given.</summary>
    /// <param name="settings">The parsed command line.</param>
    /// <param name="cancellationToken">Cancels the request.</param>
    /// <returns>The process exit code.</returns>
    public static async Task<int> ClassifyAsync(VisionSettings settings, CancellationToken cancellationToken)
    {
        var image = ReadImage(settings);
        var services = new ServiceCollection().AddSharplight();
        using var device = OpenDevice(settings);
        Action<TaskLoadOptions> configure = options => Configure(options, settings, device);
        if (settings.Labels.Count > 0)
        {
            services.AddImageEmbedder(configure);
        }
        else
        {
            services.AddImageClassifier(configure);
        }

        await using var container = services.BuildServiceProvider();
        var result = settings.Labels.Count > 0
            ? await container.GetRequiredService<IImageEmbedder>().ClassifyZeroShotAsync(image, settings.Labels, new ImageClassificationOptions { TopK = settings.TopK }, cancellationToken)
            : await container.GetRequiredService<IImageClassifier>().ClassifyAsync(image, new ImageClassificationOptions { TopK = settings.TopK }, cancellationToken);
        foreach (var prediction in result.Predictions)
        {
            Console.WriteLine($"{prediction.Score,7:P1}  {prediction.Label}");
        }

        return ExitCodes.Success;
    }

    /// <summary>Detects objects, prints them and draws the boxes into the output image.</summary>
    /// <param name="settings">The parsed command line.</param>
    /// <param name="cancellationToken">Cancels the request.</param>
    /// <returns>The process exit code.</returns>
    public static async Task<int> DetectAsync(VisionSettings settings, CancellationToken cancellationToken)
    {
        var image = ReadImage(settings);
        var services = new ServiceCollection().AddSharplight();
        using var device = OpenDevice(settings);
        services.AddObjectDetector(options => Configure(options, settings, device));
        await using var container = services.BuildServiceProvider();
        var detector = container.GetRequiredService<IObjectDetector>();
        var result = await detector.DetectAsync(image, new ObjectDetectionOptions { ScoreThreshold = settings.Threshold }, cancellationToken);
        foreach (var found in result.Objects)
        {
            Console.WriteLine($"{found.Score,7:P1}  {found.Label,-16} [{found.Box.X1:F0}, {found.Box.Y1:F0}, {found.Box.X2:F0}, {found.Box.Y2:F0}]");
        }

        if (settings.OutputPath is not null)
        {
            var annotated = new ImageAnnotator(image);
            foreach (var found in result.Objects)
            {
                annotated.DrawBox(found.Box, found.ClassIndex, $"{found.Label} {found.Score:P0}");
            }

            annotated.Save(settings.OutputPath);
        }

        return ExitCodes.Success;
    }

    private static ImageInput ReadImage(VisionSettings settings)
        => File.Exists(settings.ImagePath) ? PngCodec.Read(settings.ImagePath) : throw new SampleException($"Image '{settings.ImagePath}' not found.", ExitCodes.InvalidInput);

    private static IComputeDevice OpenDevice(VisionSettings settings)
    {
        var provider = SampleRuntime.SelectProvider(settings.UseCpu);
        return provider.OpenDevice(SampleRuntime.GetDevice(provider, settings.DeviceIndex));
    }

    private static void Configure(TaskLoadOptions options, VisionSettings settings, IComputeDevice device)
    {
        options.ModelPath = settings.ModelPath;
        options.Devices = [new DeviceSlot { Device = device }];
        options.Log = settings.Verbose ? message => Console.Error.WriteLine(message) : null;
    }
}
