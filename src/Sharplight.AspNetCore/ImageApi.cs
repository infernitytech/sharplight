using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.DependencyInjection;
using Sharplight.ImageGeneration;
using Sharplight.Imaging;

namespace Sharplight.AspNetCore;

/// <summary>
/// OpenAI-compatible image generation: <c>POST /v1/images/generations</c> with <c>prompt</c>, <c>n</c>, <c>size</c> ("512x512") and
/// <c>response_format</c> <c>b64_json</c> (PNG; <c>url</c> is not supported), plus the extensions <c>negative_prompt</c>,
/// <c>steps</c>, <c>guidance_scale</c>, <c>seed</c> and <c>sampler</c>. Generators are registered with
/// <see cref="AddSharplightImageGenerator"/>; <c>MapSharplight</c> maps the endpoint when one is registered. Requests to one generator
/// run one at a time.
/// </summary>
/// <example>
/// <code>
/// builder.Services.AddImageGenerator(o => { o.ModelPath = "sd15.slm"; o.Devices = [slot]; });
/// builder.Services.AddSharplightImageGenerator(name: "sd15");   // the container's IImageGenerator
/// var app = builder.Build();
/// app.UseSharplight();
/// app.MapSharplightImages();   // POST /v1/images/generations with {"model": "sd15", "prompt": "a lighthouse"}
/// </code>
/// </example>
public static class ImageApi
{
    private static readonly Dictionary<IImageGenerator, SemaphoreSlim> s_gates = new(ReferenceEqualityComparer.Instance);

    extension(IServiceCollection services)
    {
        /// <summary>
        /// Serves an image generator on <c>/v1/images/generations</c> under <paramref name="name"/>: <paramref name="generator"/>, or the
        /// container's <see cref="IImageGenerator"/> when it is <see langword="null"/> (resolved when the endpoint is mapped).
        /// </summary>
        public IServiceCollection AddSharplightImageGenerator(IImageGenerator? generator = null, string name = "default", Action<ImageModels>? configure = null)
        {
            ArgumentNullException.ThrowIfNull(services);
            var models = services.FirstOrDefault(d => d.ServiceType == typeof(ImageModels))?.ImplementationInstance as ImageModels;
            if (models is null)
            {
                models = new ImageModels();
                services.AddSingleton(models);
            }

            if (generator is not null)
            {
                models.Generators[name] = generator;
            }
            else
            {
                services.AddSingleton(new PendingImageGenerator(name));
            }

            models.DefaultGenerator ??= name;
            configure?.Invoke(models);
            return services;
        }
    }

    extension(IEndpointRouteBuilder routes)
    {
        /// <summary>Maps <c>/v1/images/generations</c> for the registered generators (nothing when none are registered).</summary>
        public IEndpointRouteBuilder MapSharplightImages()
        {
            ArgumentNullException.ThrowIfNull(routes);
            if (routes.ServiceProvider.GetService<ImageModels>() is not { } models)
            {
                return routes;
            }

            foreach (var pending in routes.ServiceProvider.GetServices<PendingImageGenerator>())
            {
                models.Generators[pending.Name] = routes.ServiceProvider.GetRequiredService<IImageGenerator>();
            }

            if (models.Generators.Count > 0)
            {
                routes.MapPost("/v1/images/generations", http => Handle(http, () => Generate(http, models)));
            }

            return routes;
        }
    }

    private static async Task Handle(HttpContext http, Func<Task> body)
    {
        try
        {
            await body();
        }
        catch (ApiException ex)
        {
            await ApiErrors.WriteAsync(http, ex.Status, ex.Type, ex.Message, ex.Code, ex.Param);
        }
        catch (Exception ex) when (ex is ArgumentException or NotSupportedException)
        {
            await ApiErrors.WriteAsync(http, 400, "invalid_request_error", ex.Message);
        }
    }

    private static async Task Generate(HttpContext http, ImageModels models)
    {
        using var document = await HttpJson.ReadBodyAsync(http, models.MaxRequestBodyBytes);
        var body = document.RootElement;
        string prompt = body.Str("prompt") is { Length: > 0 } p ? p : throw ApiException.BadRequest("'prompt' is required.", "prompt");
        var generator = Pick(models, body.Str("model"));
        if (body.Str("response_format") is { } format && format != "b64_json")
        {
            throw ApiException.BadRequest("Only response_format \"b64_json\" is supported.", "response_format");
        }

        int count = body.Int("n") ?? 1;
        if (count < 1 || count > models.MaxImages)
        {
            throw ApiException.BadRequest($"'n' must be between 1 and {models.MaxImages}.", "n");
        }

        var (width, height) = ParseSize(body.Str("size"));
        var request = new ImageGenerationRequest { Prompt = prompt, NegativePrompt = body.Str("negative_prompt") };
        var options = new ImageGenerationOptions
        {
            Width = width,
            Height = height,
            Count = count,
            Steps = body.Int("steps"),
            GuidanceScale = body.Float("guidance_scale"),
            Seed = body.Get("seed") is { ValueKind: JsonValueKind.Number } seed ? seed.GetInt64() : null,
            Scheduler = body.Str("sampler"),
        };
        SemaphoreSlim gate;
        lock (s_gates)
        {
            if (!s_gates.TryGetValue(generator, out gate!))
            {
                gate = new SemaphoreSlim(1, 1);
                s_gates[generator] = gate;
            }
        }

        await gate.WaitAsync(http.RequestAborted);
        ImageGenerationResult result;
        try
        {
            result = await generator.GenerateAsync(request, options, http.RequestAborted);
        }
        finally
        {
            gate.Release();
        }

        var data = new JsonArray();
        foreach (var image in result.Images)
        {
            data.Add(new JsonObject { ["b64_json"] = Convert.ToBase64String(PngCodec.Encode(image)) });
        }

        await HttpJson.WriteAsync(http, new JsonObject
        {
            ["created"] = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
            ["data"] = data,
            ["seed"] = result.Seed,
        });
    }

    private static IImageGenerator Pick(ImageModels models, string? requested)
    {
        if (requested is not null && models.Generators.TryGetValue(requested, out var named))
        {
            return named;
        }

        // A single model answers to any name (clients often hard-code e.g. "dall-e-2").
        if (requested is null || models.Generators.Count == 1)
        {
            return models.Generators[models.DefaultGenerator ?? models.Generators.Keys.First()];
        }

        throw new ApiException(404, "invalid_request_error", $"The image model '{requested}' does not exist (available: {string.Join(", ", models.Generators.Keys)}).",
            "model_not_found", "model");
    }

    /// <summary>OpenAI's <c>size</c> ("WIDTHxHEIGHT", or "auto" / absent for the model's native size).</summary>
    private static (int? Width, int? Height) ParseSize(string? size)
    {
        if (size is null or "auto")
        {
            return (null, null);
        }

        var parts = size.Split('x');
        return parts.Length == 2 && int.TryParse(parts[0], NumberStyles.None, CultureInfo.InvariantCulture, out int width)
               && int.TryParse(parts[1], NumberStyles.None, CultureInfo.InvariantCulture, out int height) && width > 0 && height > 0
            ? (width, height)
            : throw ApiException.BadRequest($"'size' must be WIDTHxHEIGHT, got '{size}'.", "size");
    }
}
