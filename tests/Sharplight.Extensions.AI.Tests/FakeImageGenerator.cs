using System.Runtime.CompilerServices;
using Sharplight.ImageGeneration;
using Sharplight.Models;
using Sharplight.Tasks;
using Sharplight.Vision;

namespace Sharplight.Extensions.AI.Tests;

/// <summary>An image generator that paints every pixel with the request's seed.</summary>
internal sealed class FakeImageGenerator : IImageGenerator
{
    public ImageGenerationRequest? LastRequest { get; private set; }

    public ImageGenerationOptions? LastOptions { get; private set; }

    public TaskClientMetadata Metadata { get; } = new(ModelKinds.ImageGeneration, "fake-sd", "fake");

    public object? GetService(Type serviceType, object? serviceKey = null) => null;

    public Task<ImageGenerationResult> GenerateAsync(ImageGenerationRequest request, ImageGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        LastRequest = request;
        LastOptions = options;
        int width = options?.Width ?? 8;
        int height = options?.Height ?? 8;
        long seed = options?.Seed ?? 1;
        var images = Enumerable.Range(0, options?.Count ?? 1)
            .Select(i => new ImageInput(width, height, Enumerable.Repeat((byte)(seed + i), width * height * 3).ToArray()))
            .ToList();
        return Task.FromResult(new ImageGenerationResult(images, seed));
    }

    public async IAsyncEnumerable<ImageGenerationUpdate> GenerateStreamingAsync(ImageGenerationRequest request, ImageGenerationOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        yield return new ImageGenerationUpdate(1, 1, await GenerateAsync(request, options, cancellationToken));
    }

    public void Dispose()
    {
    }
}
