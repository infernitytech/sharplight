using Microsoft.Extensions.DependencyInjection;
using Sharplight.Imaging;
using Sharplight.Runtime;
using Sharplight.Samples;
using Sharplight.Tasks;

namespace Sharplight.Vision.Sample;

/// <summary>
/// Runs <c>sharplight-vision</c>: registers Sharplight and one task client for the model with the <see cref="SampleRuntime"/> container (the
/// model loads on first use), reads the PNG, runs the task and prints the result; detections and masks are also drawn into the output image, an upscaled image is the output.
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
        Action<TaskLoadOptions, IServiceProvider> configure = (options, services) => Configure(options, settings, services);
        using var runtime = new SampleRuntime(services =>
        {
            if (settings.Labels.Count > 0)
            {
                services.AddImageEmbedder(configure);
            }
            else
            {
                services.AddImageClassifier(configure);
            }
        });
        var container = runtime.Services;
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
        using var runtime = new SampleRuntime(services => services.AddObjectDetector((options, container) => Configure(options, settings, container)));
        var container = runtime.Services;
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

    /// <summary>Estimates the depth map and writes it as a grayscale PNG (bright is near).</summary>
    /// <param name="settings">The parsed command line.</param>
    /// <param name="cancellationToken">Cancels the request.</param>
    /// <returns>The process exit code.</returns>
    public static async Task<int> DepthAsync(VisionSettings settings, CancellationToken cancellationToken)
    {
        var image = ReadImage(settings);
        using var runtime = new SampleRuntime(services => services.AddDepthEstimator((options, container) => Configure(options, settings, container)));
        var container = runtime.Services;
        var depth = await container.GetRequiredService<IDepthEstimator>().EstimateAsync(image, null, cancellationToken);
        Console.WriteLine($"depth map {depth.Width}x{depth.Height}, values {depth.Min:F3} .. {depth.Max:F3}");
        if (settings.OutputPath is not null)
        {
            var gray = depth.ToGrayscale();
            var rgb = new byte[gray.Length * 3];
            for (int i = 0; i < gray.Length; i++)
            {
                rgb[3 * i] = rgb[3 * i + 1] = rgb[3 * i + 2] = gray[i];
            }

            PngCodec.Write(settings.OutputPath, new ImageInput(depth.Width, depth.Height, rgb));
        }

        return ExitCodes.Success;
    }

    /// <summary>Segments the object at the points or box, prints the candidate masks and draws the best one into the output image.</summary>
    /// <param name="settings">The parsed command line.</param>
    /// <param name="cancellationToken">Cancels the request.</param>
    /// <returns>The process exit code.</returns>
    public static async Task<int> SegmentAsync(VisionSettings settings, CancellationToken cancellationToken)
    {
        var image = ReadImage(settings);
        var points = settings.Points.Select(text => Numbers(text, 2, "--point")).Select(v => new SegmentationPoint(v[0], v[1])).Concat(
            settings.NegativePoints.Select(text => Numbers(text, 2, "--not")).Select(v => new SegmentationPoint(v[0], v[1], false))).ToList();
        BoundingBox? box = null;
        if (settings.Box is not null)
        {
            var corners = Numbers(settings.Box, 4, "--box");
            box = new BoundingBox(corners[0], corners[1], corners[2], corners[3]);
        }

        if (points.Count == 0 && box is null)
        {
            throw new SampleException("Give at least one --point or a --box.", ExitCodes.InvalidInput);
        }

        using var runtime = new SampleRuntime(services => services.AddImageSegmenter((options, container) => Configure(options, settings, container)));
        var segmenter = runtime.Services.GetRequiredService<IImageSegmenter>();
        var result = await segmenter.SegmentAsync(image, new SegmentationPrompt { Points = points, Box = box }, null, cancellationToken);
        for (int i = 0; i < result.Masks.Count; i++)
        {
            Console.WriteLine($"mask {i + 1}: predicted IoU {result.Masks[i].Score:P1}, {result.Masks[i].Area} pixels");
        }

        Console.WriteLine($"object score {result.ObjectScore:F2}");
        if (settings.OutputPath is not null)
        {
            var annotated = new ImageAnnotator(image);
            annotated.DrawMask(result.Best.Pixels.Select(p => p ? (byte)1 : (byte)0).ToArray(), 3);
            foreach (var point in points)
            {
                annotated.DrawPoint((int)point.X, (int)point.Y, Math.Max(3, image.Width / 160), point.IsPositive ? 1 : 0);
            }

            if (box is { } outline)
            {
                annotated.DrawBox(outline, 2, "prompt");
            }

            annotated.Save(settings.OutputPath);
        }

        return ExitCodes.Success;
    }

    /// <summary>Reads the text of the image and prints it.</summary>
    /// <param name="settings">The parsed command line.</param>
    /// <param name="cancellationToken">Cancels the request.</param>
    /// <returns>The process exit code.</returns>
    public static async Task<int> OcrAsync(VisionSettings settings, CancellationToken cancellationToken)
    {
        var image = ReadImage(settings);
        using var runtime = new SampleRuntime(services => services.AddTextRecognizer((options, container) => Configure(options, settings, container)));
        var result = await runtime.Services.GetRequiredService<ITextRecognizer>().RecognizeAsync(image, null, cancellationToken);
        Console.WriteLine(result.Text);
        return ExitCodes.Success;
    }

    /// <summary>Enlarges the image and writes it as PNG to the output path.</summary>
    /// <param name="settings">The parsed command line.</param>
    /// <param name="cancellationToken">Cancels the request.</param>
    /// <returns>The process exit code.</returns>
    public static async Task<int> UpscaleAsync(VisionSettings settings, CancellationToken cancellationToken)
    {
        var image = ReadImage(settings);
        using var runtime = new SampleRuntime(services => services.AddImageUpscaler((options, container) => Configure(options, settings, container)));
        var upscaler = runtime.Services.GetRequiredService<IImageUpscaler>();
        var progress = new Progress<double>(fraction => Console.Error.Write($"\rtiles {fraction:P0}"));
        var result = await upscaler.UpscaleAsync(image, new ImageUpscalingOptions { TileSize = settings.TileSize, TilePadding = settings.TilePadding }, progress, cancellationToken);
        Console.Error.WriteLine();
        Console.WriteLine($"{image.Width}x{image.Height} -> {result.Width}x{result.Height} ({upscaler.Scale}x)");
        if (settings.OutputPath is not null)
        {
            PngCodec.Write(settings.OutputPath, result);
        }

        return ExitCodes.Success;
    }

    private static float[] Numbers(string text, int count, string option)
    {
        var parts = text.Split(',', StringSplitOptions.TrimEntries);
        if (parts.Length != count || parts.Any(p => !float.TryParse(p, System.Globalization.CultureInfo.InvariantCulture, out _)))
        {
            throw new SampleException($"{option} takes {count} numbers separated by commas, got '{text}'.", ExitCodes.InvalidInput);
        }

        return [.. parts.Select(p => float.Parse(p, System.Globalization.CultureInfo.InvariantCulture))];
    }

    private static ImageInput ReadImage(VisionSettings settings)
        => File.Exists(settings.ImagePath) ? PngCodec.Read(settings.ImagePath) : throw new SampleException($"Image '{settings.ImagePath}' not found.", ExitCodes.InvalidInput);

    /// <summary>
    /// Points the task client at the model and opens its device from the container's providers when the client is first resolved; the
    /// device lives until the process exits.
    /// </summary>
    private static void Configure(TaskLoadOptions options, VisionSettings settings, IServiceProvider services)
    {
        var provider = SampleRuntime.SelectProvider(services.GetRequiredService<ExecutionProviderCatalog>(), settings.UseCpu);
        var device = provider.OpenDevice(SampleRuntime.GetDevice(provider, settings.DeviceIndex));
        options.ModelPath = settings.ModelPath;
        options.Devices = [new DeviceSlot { Device = device }];
        options.Log = settings.Verbose ? message => Console.Error.WriteLine(message) : null;
    }
}
