using System.Runtime.CompilerServices;
using Sharplight.ImageGeneration;
using Sharplight.Tasks;
using Sharplight.Vision;

namespace Sharplight.Studio.Tests.Fakes;

/// <summary>An image generator that streams scripted steps with previews and returns gradient images; records what it was asked.</summary>
internal sealed class FakeImageGenerator : IImageGenerator
{
    public TaskClientMetadata Metadata { get; } = new(Sharplight.Models.ModelKinds.ImageGeneration, "fake-sd", "latent_diffusion");

    /// <summary>The requests and options received.</summary>
    public List<(ImageGenerationRequest Request, ImageGenerationOptions Options)> Requests { get; } = [];

    /// <summary>The size of the images when the options name none.</summary>
    public int NativeSize { get; set; } = 64;

    /// <summary>When set, the stream waits on it after the first step (until cancelled).</summary>
    public TaskCompletionSource? Gate { get; set; }

    /// <summary>Blocking work per step (a stand-in for a graph run that holds its thread), or zero.</summary>
    public TimeSpan StepWork { get; set; }

    public Task<ImageGenerationResult> GenerateAsync(ImageGenerationRequest request, ImageGenerationOptions? options = null, CancellationToken cancellationToken = default)
        => throw new NotSupportedException("The page streams.");

    public async IAsyncEnumerable<ImageGenerationUpdate> GenerateStreamingAsync(ImageGenerationRequest request, ImageGenerationOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        options ??= ImageGenerationOptions.Default;
        Requests.Add((request, options));
        int steps = options.Steps ?? 4;
        for (int step = 1; step <= steps; step++)
        {
            await Task.Yield();
            cancellationToken.ThrowIfCancellationRequested();
            SlowWork.Block(StepWork);
            var preview = options.PreviewInterval > 0 && step % options.PreviewInterval == 0 ? Gradient(8, 8, step) : null;
            yield return new ImageGenerationUpdate(step, steps, null, preview);
            if (step == 1 && Gate is { } gate)
            {
                await gate.Task.WaitAsync(cancellationToken);
            }
        }

        long seed = options.Seed ?? 424242;
        int width = options.Width ?? NativeSize;
        int height = options.Height ?? NativeSize;
        var images = Enumerable.Range(0, options.Count).Select(i => Gradient(width, height, (int)(seed + i))).ToList();
        yield return new ImageGenerationUpdate(steps, steps, new ImageGenerationResult(images, seed));
    }

    public object? GetService(Type serviceType, object? serviceKey = null) => null;

    public void Dispose()
    {
    }

    internal static ImageInput Gradient(int width, int height, int tint)
    {
        var rgb = new byte[3 * width * height];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int i = 3 * (y * width + x);
                rgb[i] = (byte)(x * 255 / Math.Max(1, width - 1));
                rgb[i + 1] = (byte)(y * 255 / Math.Max(1, height - 1));
                rgb[i + 2] = (byte)(tint * 37);
            }
        }

        return new ImageInput(width, height, rgb);
    }
}
