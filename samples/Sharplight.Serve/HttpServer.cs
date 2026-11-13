using Sharplight.Llm.Serving;
using Sharplight.Llm.Serving.Hosting;
using Sharplight.Samples;
using Sharplight.Server;

namespace Sharplight.Serve;

/// <summary>Hosts the OpenAI and/or Anthropic endpoints for a <see cref="ModelHost"/> on Kestrel.</summary>
internal static class HttpServer
{
    /// <summary>Serves until <paramref name="cancellationToken"/> is cancelled (Ctrl+C).</summary>
    public static async Task RunAsync(ModelHost host, ServeSettings settings, ServerPlan plan, CancellationToken cancellationToken)
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
        builder.Services.AddSharplight(host, options =>
        {
            options.OpenAi = settings.Api is ApiSurface.OpenAi or ApiSurface.Both;
            options.Anthropic = settings.Api is ApiSurface.Anthropic or ApiSurface.Both;
            options.ApiKeys.AddRange(settings.ApiKeys);
            options.CorsOrigins.AddRange(settings.CorsOrigins);
            options.DefaultSampling = new SamplingParams { Temperature = settings.DefaultTemperature };

            // A single model answers to any name (clients often send arbitrary ones); with several, an unknown name is a 404.
            options.UnknownModelFallsBackToDefault = plan.Models.Count == 1;
        });

        var app = builder.Build();
        app.UseSharplight();
        app.MapSharplight();
        PrintBanner(host, settings, plan, builder.WebHost.GetSetting("urls"));
        await app.RunAsync(cancellationToken);
    }

    private static void PrintBanner(ModelHost host, ServeSettings settings, ServerPlan plan, string? urls)
    {
        var names = string.Join(", ", plan.Models.Select(model => $"'{model.Name}'"));
        var limit = plan.MaxLoadedModels > 0 ? $", max {plan.MaxLoadedModels} loaded" : "";
        var api = settings.Api.ToString().ToLowerInvariant();
        Console.Error.WriteLine($"Serving {names} ({api} API, VRAM reserve {ByteSize.Format(plan.ReserveBytes)}{limit}) on {urls}");
        Console.Error.WriteLine(host.Describe());
    }
}
