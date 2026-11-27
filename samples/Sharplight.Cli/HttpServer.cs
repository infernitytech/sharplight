using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Sharplight.AspNetCore;
using Sharplight.Hosting;
using Sharplight.Llm.Serving;
using Sharplight.Llm.Serving.Hosting;
using Sharplight.Samples;

namespace Sharplight.Cli;

/// <summary>Hosts the models with Sharplight.Hosting and serves them over the OpenAI and/or Anthropic endpoints of Sharplight.AspNetCore on Kestrel.</summary>
internal static class HttpServer
{
    /// <summary>Serves until <paramref name="cancellationToken"/> is cancelled (Ctrl+C).</summary>
    public static async Task RunAsync(SampleRuntime runtime, ModelHostingOptions hosting, ServeSettings settings, CancellationToken cancellationToken)
    {
        // No configuration file watching: the server is configured by its command line, and watching the working directory
        // costs CPU on every change there.
        var builder = WebApplication.CreateBuilder(new WebApplicationOptions
        {
            Args = ["--hostBuilder:reloadConfigOnChange=false"],
            ContentRootPath = AppContext.BaseDirectory,
        });
        builder.WebHost.UseUrls(settings.ListenUrl);

        // Prompts can be large; the models' own context limits bound the request size.
        builder.WebHost.ConfigureKestrel(kestrel => kestrel.Limits.MaxRequestBodySize = null);
        builder.Logging.SetMinimumLevel(LogLevel.Warning);
        runtime.AddSharedServices(builder.Services);
        if (hosting.OnDemand)
        {
            builder.Services.AddSingleton<IHostedModelResolver>(new StoreModelResolver(settings.StoreRoot));
        }

        // With several models the load lines need a prefix to tell them apart, and several may load at once (lazily, while serving):
        // then plain progress lines, not one updating line.
        bool prefixLog = hosting.Lazy || hosting.OnDemand || hosting.Models.Count > 1;
        builder.Services.AddSharplightHosting(options =>
        {
            ModelHostingJson.CopyTo(hosting, options);
            options.Log = Console.Error.WriteLine;
            options.ConfigureLoad = (model, load) =>
            {
                var progress = prefixLog ? new ConsoleLoadProgress($"[{model.Name}] ", live: false) : new ConsoleLoadProgress();
                load.Log = progress.Log;
                load.Progress = progress;
            };
        });
        builder.Services.AddSharplightServer(options =>
        {
            options.OpenAi = settings.Api is ApiSurface.OpenAi or ApiSurface.Both;
            options.Anthropic = settings.Api is ApiSurface.Anthropic or ApiSurface.Both;
            foreach (var key in settings.ApiKeys)
            {
                options.ApiKeys.Add(key);
            }

            foreach (var origin in settings.CorsOrigins)
            {
                options.CorsOrigins.Add(origin);
            }

            options.DefaultSampling = new SamplingParams { Temperature = settings.DefaultTemperature };
            options.AllowClientCacheSalt = settings.AllowClientCacheSalt;

            // A single chat model answers to any name (clients often send arbitrary ones); with several, an unknown name is a 404.
            // Task models (speech, images, time-series forecasting) are served from the host per request by their name.
            options.ServeHostModels = hosting.OnDemand || hosting.Models.Any(model => model.Kind is ModelKind.Task);
            options.UnknownModelFallsBackToDefault = hosting.Models.Count(model => model.Kind == ModelKind.Chat) == 1 && !hosting.OnDemand;
        });

        await using var app = builder.Build();
        app.UseSharplightServer();
        app.MapSharplightServer();
        try
        {
            await app.StartAsync(cancellationToken);
        }
        catch (Exception error) when (error is not OperationCanceledException)
        {
            throw new SampleException($"Cannot load model: {error.Message}", error);
        }

        PrintBanner(app.Services.GetRequiredService<IModelHostManager>(), hosting, settings, string.Join(", ", app.Urls));
        await app.WaitForShutdownAsync(cancellationToken);
    }

    private static void PrintBanner(IModelHostManager manager, ModelHostingOptions hosting, ServeSettings settings, string urls)
    {
        var names = string.Join(", ", hosting.Models.Select(model => $"'{model.Name}'"));
        var limit = hosting.MaxLoadedModels > 0 ? $", max {hosting.MaxLoadedModels} loaded" : "";
        var api = settings.Api.ToString().ToLowerInvariant();
        Console.Error.WriteLine($"Serving {names} ({api} API, VRAM reserve {MemorySize.Format(hosting.Reserve)}{limit}) on {urls}");
        Console.Error.WriteLine(manager.Host.Describe());
    }
}
