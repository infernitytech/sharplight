using System.Globalization;
using Microsoft.Extensions.AI;
using Sharplight.Imaging;
using MeaiImageGenerationOptions = Microsoft.Extensions.AI.ImageGenerationOptions;
using MeaiImageGenerationRequest = Microsoft.Extensions.AI.ImageGenerationRequest;
using MeaiImageGenerator = Microsoft.Extensions.AI.IImageGenerator;
using SharplightImageGenerationOptions = Sharplight.ImageGeneration.ImageGenerationOptions;
using SharplightImageGenerationRequest = Sharplight.ImageGeneration.ImageGenerationRequest;
using SharplightImageGeneratorClient = Sharplight.ImageGeneration.IImageGenerator;

namespace Sharplight.Extensions.AI;

/// <summary>
/// Microsoft.Extensions.AI <c>IImageGenerator</c> over a Sharplight image generator. Images come back as PNG <see cref="DataContent"/>.
/// The first image of <see cref="MeaiImageGenerationRequest.OriginalImages"/> (PNG, JPEG, ...) makes it an image-to-image request.
/// <see cref="MeaiImageGenerationOptions.Count"/> and <see cref="MeaiImageGenerationOptions.ImageSize"/> map directly;
/// <see cref="MeaiImageGenerationOptions.AdditionalProperties"/> may carry <c>negative_prompt</c>, <c>steps</c>, <c>guidance_scale</c>,
/// <c>seed</c>, <c>strength</c> and <c>scheduler</c>.
/// </summary>
public sealed class SharplightImageGenerator : MeaiImageGenerator
{
    private const string PngMediaType = "image/png";
    private readonly SharplightImageGeneratorClient _generator;
    private readonly bool _ownsGenerator;
    private readonly ImageGeneratorMetadata _metadata;

    /// <summary>Wraps <paramref name="generator"/>; disposes it with the adapter when <paramref name="ownsGenerator"/>.</summary>
    /// <param name="generator">The Sharplight image generator.</param>
    /// <param name="ownsGenerator">Whether disposing the adapter disposes the generator.</param>
    public SharplightImageGenerator(SharplightImageGeneratorClient generator, bool ownsGenerator = false)
    {
        ArgumentNullException.ThrowIfNull(generator);
        _generator = generator;
        _ownsGenerator = ownsGenerator;
        _metadata = new ImageGeneratorMetadata(TaskClientMetadataNames.Provider, null, generator.Metadata.ModelName);
    }

    /// <inheritdoc/>
    /// <exception cref="NotSupportedException">A media type other than PNG is requested.</exception>
    public async Task<ImageGenerationResponse> GenerateAsync(MeaiImageGenerationRequest request, MeaiImageGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (options?.MediaType is { } mediaType && mediaType != PngMediaType)
        {
            throw new NotSupportedException($"Media type '{mediaType}' is not supported; Sharplight returns {PngMediaType}.");
        }

        var result = await _generator.GenerateAsync(ToRequest(request, options), ToOptions(options), cancellationToken).ConfigureAwait(false);
        var contents = result.Images.Select(image => (AIContent)new DataContent(PngCodec.Encode(image), PngMediaType)).ToList();
        return new ImageGenerationResponse(contents) { RawRepresentation = result };
    }

    /// <inheritdoc/>
    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        if (serviceKey is not null)
        {
            return null;
        }

        if (serviceType == typeof(ImageGeneratorMetadata))
        {
            return _metadata;
        }

        return serviceType.IsInstanceOfType(this) ? this : serviceType.IsInstanceOfType(_generator) ? _generator : _generator.GetService(serviceType);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_ownsGenerator)
        {
            _generator.Dispose();
        }
    }

    private static SharplightImageGenerationRequest ToRequest(MeaiImageGenerationRequest request, MeaiImageGenerationOptions? options)
    {
        var original = request.OriginalImages?.OfType<DataContent>().FirstOrDefault();
        return new SharplightImageGenerationRequest
        {
            Prompt = request.Prompt ?? "",
            NegativePrompt = Property<string>(options, "negative_prompt"),
            Image = original is null ? null : ImageDecoding.DecodeWithStb(original),
            Strength = Property<float?>(options, "strength") ?? 0.75f,
        };
    }

    private static SharplightImageGenerationOptions ToOptions(MeaiImageGenerationOptions? options)
    {
        if (options is null)
        {
            return SharplightImageGenerationOptions.Default;
        }

        return new SharplightImageGenerationOptions
        {
            Width = options.ImageSize?.Width,
            Height = options.ImageSize?.Height,
            Count = options.Count ?? 1,
            Steps = Property<int?>(options, "steps"),
            GuidanceScale = Property<float?>(options, "guidance_scale"),
            Seed = Property<long?>(options, "seed"),
            Scheduler = Property<string>(options, "scheduler"),
        };
    }

    private static T? Property<T>(MeaiImageGenerationOptions? options, string key)
    {
        if (options?.AdditionalProperties is not { } properties || !properties.TryGetValue(key, out var value) || value is null)
        {
            return default;
        }

        var target = Nullable.GetUnderlyingType(typeof(T)) ?? typeof(T);
        return (T)Convert.ChangeType(value is System.Text.Json.JsonElement json ? json.ToString() : value, target, CultureInfo.InvariantCulture);
    }
}
