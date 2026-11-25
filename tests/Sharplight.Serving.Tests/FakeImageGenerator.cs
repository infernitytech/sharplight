using System.Runtime.CompilerServices;
using Sharplight.ImageGeneration;
using Sharplight.Tasks;
using Sharplight.Vision;

namespace Sharplight.Serving.Tests;

/// <summary>Image generator that returns solid images whose red channel is the seed (mod 256); records the requests.</summary>
internal sealed class FakeImageGenerator : IImageGenerator
{
    public TaskClientMetadata Metadata { get; } = new("image-generation", "fake-sd", "latent_diffusion");

    public ImageGenerationDefaults Defaults { get; } = new(64, 64, 20, 7.5f, "euler");

    public List<(ImageGenerationRequest Request, ImageGenerationOptions Options)> Calls { get; } = [];

    public Task<ImageGenerationResult> GenerateAsync(ImageGenerationRequest request, ImageGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new ImageGenerationOptions();
        Calls.Add((request, options));
        long seed = options.Seed ?? 1;
        int width = options.Width ?? 8;
        int height = options.Height ?? 8;
        var images = new List<ImageInput>();
        for (int i = 0; i < options.Count; i++)
        {
            var rgb = new byte[width * height * 3];
            for (int p = 0; p < rgb.Length; p += 3)
            {
                rgb[p] = (byte)((seed + i) % 256);
            }

            images.Add(new ImageInput(width, height, rgb));
        }

        return Task.FromResult(new ImageGenerationResult(images, seed));
    }

    public async IAsyncEnumerable<ImageGenerationUpdate> GenerateStreamingAsync(ImageGenerationRequest request, ImageGenerationOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var result = await GenerateAsync(request, options, cancellationToken);
        yield return new ImageGenerationUpdate(1, 1, result);
    }

    public object? GetService(Type serviceType, object? serviceKey = null) => serviceType.IsInstanceOfType(this) ? this : null;

    public void Dispose()
    {
    }
}
