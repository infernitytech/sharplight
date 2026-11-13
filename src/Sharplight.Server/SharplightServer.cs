using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Sharplight.Llm.Serving;
using Sharplight.Llm.Serving.Chat;

namespace Sharplight.Server;

public sealed class SharplightServerOptions
{
    /// <summary>Expose the OpenAI-compatible API (<c>/v1/chat/completions</c>, <c>/v1/completions</c>, <c>/v1/models</c>).</summary>
    public bool OpenAi { get; set; } = true;
    /// <summary>Expose the Anthropic-compatible API (<c>/v1/messages</c>, <c>/v1/messages/count_tokens</c>).</summary>
    public bool Anthropic { get; set; } = true;
    /// <summary>Accepted API keys (<c>Authorization: Bearer</c> or <c>x-api-key</c>). Empty = no authentication.</summary>
    public List<string> ApiKeys { get; set; } = [];
    /// <summary>Allowed CORS origins ("*" for any). Empty = CORS headers are not sent.</summary>
    public List<string> CorsOrigins { get; set; } = [];
    /// <summary>Largest accepted request body (images arrive base64 encoded in the body).</summary>
    public long MaxRequestBodyBytes { get; set; } = 32 * 1024 * 1024;
    /// <summary>Largest <c>n</c> a request may ask for.</summary>
    public int MaxChoices { get; set; } = 8;
    /// <summary>Total time budget per request (queue + generation); null = none.</summary>
    public TimeSpan? RequestTimeout { get; set; }
    /// <summary>Applied when a request leaves a sampling parameter out.</summary>
    public SamplingParams DefaultSampling { get; set; } = new();
    /// <summary>How long shutdown waits for running requests to finish.</summary>
    public TimeSpan DrainTimeout { get; set; } = TimeSpan.FromSeconds(30);
    /// <summary>Serve <c>/metrics</c> (Prometheus text format).</summary>
    public bool Metrics { get; set; } = true;
}

/// <summary>Shared state behind the endpoints (one per host).</summary>
public sealed class SharplightServerContext(ModelService model, SharplightServerOptions options)
{
    public ModelService Model { get; } = model;
    public SharplightServerOptions Options { get; } = options;
    internal bool ModelsMapped;
}

/// <summary>Drains the engine when the host stops so running requests can finish while new ones get 503.</summary>
sealed class EngineLifetime(SharplightServerContext ctx) : IHostedService
{
    public Task StartAsync(CancellationToken ct) => Task.CompletedTask;
    public async Task StopAsync(CancellationToken ct) => await ctx.Model.Engine.DrainAsync(ctx.Options.DrainTimeout);
}

public static class SharplightServerExtensions
{
    /// <summary>Registers the model service and options. The caller keeps ownership of the engine (dispose it after the host stops).</summary>
    public static IServiceCollection AddSharplight(this IServiceCollection services, ModelService model, Action<SharplightServerOptions>? configure = null)
    {
        var options = new SharplightServerOptions();
        configure?.Invoke(options);
        var ctx = new SharplightServerContext(model, options);
        services.AddSingleton(ctx);
        services.AddHostedService(_ => new EngineLifetime(ctx));
        return services;
    }

    /// <summary>Adds CORS handling and API-key authentication in front of the endpoints (call before the Map* methods).</summary>
    public static IApplicationBuilder UseSharplight(this IApplicationBuilder app)
    {
        var ctx = app.ApplicationServices.GetRequiredService<SharplightServerContext>();
        app.Use(async (http, next) =>
        {
            var o = ctx.Options;
            if (o.CorsOrigins.Count > 0 && http.Request.Headers.Origin.Count > 0)
            {
                string origin = http.Request.Headers.Origin!;
                if (o.CorsOrigins.Contains("*") || o.CorsOrigins.Contains(origin, StringComparer.OrdinalIgnoreCase))
                {
                    http.Response.Headers.AccessControlAllowOrigin = o.CorsOrigins.Contains("*") ? "*" : origin;
                    http.Response.Headers.Vary = "Origin";
                    http.Response.Headers.AccessControlExposeHeaders = "*";
                }
                if (HttpMethods.IsOptions(http.Request.Method))
                {
                    http.Response.Headers.AccessControlAllowMethods = "GET, POST, OPTIONS";
                    http.Response.Headers.AccessControlAllowHeaders = http.Request.Headers.AccessControlRequestHeaders.Count > 0
                        ? http.Request.Headers.AccessControlRequestHeaders : "*";
                    http.Response.Headers.AccessControlMaxAge = "600";
                    http.Response.StatusCode = StatusCodes.Status204NoContent;
                    return;
                }
            }
            var path = http.Request.Path.Value ?? "";
            bool open = path is "/health" or "/metrics" || HttpMethods.IsOptions(http.Request.Method);
            if (o.ApiKeys.Count > 0 && !open && !KeyIsValid(http, o.ApiKeys))
            {
                await ApiErrors.WriteAsync(http, 401, "authentication_error", "Invalid or missing API key.", code: "invalid_api_key");
                return;
            }
            try { await next(http); }
            catch (ApiException ex) { await ApiErrors.WriteAsync(http, ex.Status, ex.Type, ex.Message, ex.Code, ex.Param); }
            catch (EngineException ex) { await ApiErrors.WriteAsync(http, ex); }
            catch (OperationCanceledException) when (http.RequestAborted.IsCancellationRequested) { }
            catch (Exception ex) when (!http.Response.HasStarted)
            {
                await ApiErrors.WriteAsync(http, 500, "server_error", "Internal error: " + ex.Message);
            }
        });
        return app;
    }

    static bool KeyIsValid(HttpContext http, List<string> keys)
    {
        string? presented = null;
        string auth = http.Request.Headers.Authorization.ToString();
        if (auth.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase)) presented = auth[7..].Trim();
        else if (http.Request.Headers.TryGetValue("x-api-key", out var k)) presented = k.ToString();
        if (presented is null) return false;
        var pb = System.Text.Encoding.UTF8.GetBytes(presented);
        bool ok = false;
        foreach (var key in keys)
            ok |= System.Security.Cryptography.CryptographicOperations.FixedTimeEquals(pb, System.Text.Encoding.UTF8.GetBytes(key));
        return ok;
    }

    /// <summary>Maps everything the options enable: <c>/health</c>, <c>/metrics</c>, the OpenAI and the Anthropic API.</summary>
    public static IEndpointRouteBuilder MapSharplight(this IEndpointRouteBuilder routes)
    {
        var o = routes.ServiceProvider.GetRequiredService<SharplightServerContext>().Options;
        routes.MapOperations();
        if (o.OpenAi) routes.MapOpenAi();
        if (o.Anthropic) routes.MapAnthropic();
        return routes;
    }

    /// <summary><c>GET /health</c> and (if enabled) <c>GET /metrics</c>.</summary>
    public static IEndpointRouteBuilder MapOperations(this IEndpointRouteBuilder routes)
    {
        var ctx = routes.ServiceProvider.GetRequiredService<SharplightServerContext>();
        routes.MapGet("/health", async http =>
        {
            var e = ctx.Model.Engine;
            var m = e.GetMetrics();
            http.Response.StatusCode = e.IsDraining ? 503 : 200;
            http.Response.ContentType = "application/json";
            await http.Response.WriteAsync(JsonSerializer.Serialize(new
            {
                status = e.IsDraining ? "draining" : "ok", model = ctx.Model.ModelName, slots = m.SlotCount,
                active = m.ActiveSequences, queued = m.QueueDepth,
            }));
        });
        if (ctx.Options.Metrics)
            routes.MapGet("/metrics", async http =>
            {
                http.Response.ContentType = "text/plain; version=0.0.4; charset=utf-8";
                await http.Response.WriteAsync(Prometheus.Render(ctx.Model.Engine.GetMetrics(), ctx.Model.ModelName));
            });
        return routes;
    }

    internal static void MapModels(this IEndpointRouteBuilder routes, SharplightServerContext ctx)
    {
        if (ctx.ModelsMapped) return;
        ctx.ModelsMapped = true;
        routes.MapGet("/v1/models", async http =>
        {
            bool anthropic = ApiErrors.IsAnthropic(http) && ctx.Options.Anthropic;
            await HttpJson.WriteAsync(http, anthropic ? AnthropicApi.ModelList(ctx) : OpenAiApi.ModelList(ctx));
        });
        routes.MapGet("/v1/models/{id}", async http =>
        {
            bool anthropic = ApiErrors.IsAnthropic(http) && ctx.Options.Anthropic;
            string id = (string)http.Request.RouteValues["id"]!;
            if (id != ctx.Model.ModelName)
            {
                await ApiErrors.WriteAsync(http, 404, anthropic ? "not_found_error" : "invalid_request_error", $"The model '{id}' does not exist.", code: "model_not_found");
                return;
            }
            await HttpJson.WriteAsync(http, anthropic ? AnthropicApi.ModelObject(ctx) : OpenAiApi.ModelObject(ctx));
        });
    }
}
