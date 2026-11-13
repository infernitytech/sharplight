using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Sharplight.Llm.Serving;
using Sharplight.Llm.Serving.Chat;
using Sharplight.Llm.Serving.Hosting;

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
    /// <summary>
    /// A request whose <c>model</c> names no registered model is answered by the default (first) chat model instead of a 404.
    /// On for a single-model server, where clients commonly send arbitrary names; off by default with several models.
    /// </summary>
    public bool UnknownModelFallsBackToDefault { get; set; }
    /// <summary>Serve the admin endpoints (<c>GET /admin/models</c>, <c>POST /admin/models/{name}/load|unload</c>). They need an API key when keys are configured, else a loopback caller.</summary>
    public bool Admin { get; set; } = true;
}

/// <summary>Shared state behind the endpoints (one per host): the models (a <see cref="ModelHost"/>) and the server options.</summary>
public sealed class SharplightServerContext
{
    public SharplightServerContext(ModelHost host, SharplightServerOptions options) { Host = host; Options = options; }

    /// <summary>Single-model form: the service is registered as the (pinned, never reloaded) default model. The caller keeps ownership of its engine.</summary>
    public SharplightServerContext(ModelService model, SharplightServerOptions options)
    {
        Host = new ModelHost();
        Host.Adopt(model.ModelName, ChatModelInstance.Wrap(model));
        Options = options;
        Options.UnknownModelFallsBackToDefault = true;
    }

    public ModelHost Host { get; }
    public SharplightServerOptions Options { get; }
    internal bool ModelsMapped;

    /// <summary>The default chat model's service (loads it if needed); for code written for a single model. Prefer <see cref="AcquireAsync"/>.</summary>
    public ModelService Model
    {
        get
        {
            var name = Host.DefaultName() ?? throw new InvalidOperationException("No chat model is registered.");
            using var lease = Host.AcquireAsync(name).GetAwaiter().GetResult();
            return lease.As<ChatModelInstance>().Service;   // valid until the model is unloaded; hold a lease for long work
        }
    }

    /// <summary>
    /// Resolves the request's <c>model</c> (null/empty = the default model of <paramref name="kind"/>), loading it when needed, and
    /// returns a lease that keeps it loaded until disposed (dispose after the response is written). Unknown names raise a 404 in the
    /// API's error shape, unless <see cref="SharplightServerOptions.UnknownModelFallsBackToDefault"/>.
    /// </summary>
    public async Task<ModelLease> AcquireAsync(HttpContext http, string? requested, ModelKind kind = ModelKind.Chat)
    {
        string? name = requested;
        if (string.IsNullOrEmpty(name) || (Options.UnknownModelFallsBackToDefault && !Host.Contains(name))) name = Host.DefaultName(kind);
        if (name is null) throw new ApiException(404, ApiErrors.IsAnthropic(http) ? "not_found_error" : "invalid_request_error", $"No {kind.ToString().ToLowerInvariant()} model is available.", "model_not_found", "model");
        try { return await Host.AcquireAsync(name, kind, http.RequestAborted); }
        catch (ModelNotFoundException ex) { throw new ApiException(404, ApiErrors.IsAnthropic(http) ? "not_found_error" : "invalid_request_error", ex.Message, "model_not_found", "model"); }
        catch (ModelKindMismatchException ex) { throw new ApiException(404, ApiErrors.IsAnthropic(http) ? "not_found_error" : "invalid_request_error", ex.Message, "model_not_found", "model"); }
    }

    /// <summary>Chat service of a lease from <see cref="AcquireAsync"/>.</summary>
    public static ModelService ChatOf(ModelLease lease) => lease.As<ChatModelInstance>().Service;
}

/// <summary>Drains the models when the host stops so running requests can finish while new ones get 503.</summary>
sealed class EngineLifetime(SharplightServerContext ctx) : IHostedService
{
    public Task StartAsync(CancellationToken ct) => Task.CompletedTask;
    public async Task StopAsync(CancellationToken ct) => await ctx.Host.DrainAsync(ctx.Options.DrainTimeout);
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

    /// <summary>Registers a model host (several models, loaded on demand within the device memory budget) and options. The caller disposes the host after the server stops.</summary>
    public static IServiceCollection AddSharplight(this IServiceCollection services, ModelHost host, Action<SharplightServerOptions>? configure = null)
    {
        var options = new SharplightServerOptions();
        configure?.Invoke(options);
        var ctx = new SharplightServerContext(host, options);
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
        routes.MapAdmin();
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
            var chats = ctx.LoadedChatModels();
            bool draining = chats.Count > 0 && chats.All(c => c.Instance.Engine.IsDraining);
            var def = ctx.Host.DefaultName();
            var d = chats.FirstOrDefault(c => c.Name == def);
            var m = d.Instance?.Engine.GetMetrics();
            http.Response.StatusCode = draining ? 503 : 200;
            http.Response.ContentType = "application/json";
            await http.Response.WriteAsync(JsonSerializer.Serialize(new
            {
                status = draining ? "draining" : "ok", model = def, slots = m?.SlotCount ?? 0,
                active = chats.Sum(c => c.Instance.Engine.GetMetrics().ActiveSequences), queued = chats.Sum(c => c.Instance.Engine.GetMetrics().QueueDepth),
                models = ctx.Host.List().Select(i => new { id = i.Name, kind = i.Kind.ToString().ToLowerInvariant(), state = i.State.ToString().ToLowerInvariant() }),
            }));
        });
        if (ctx.Options.Metrics)
            routes.MapGet("/metrics", async http =>
            {
                http.Response.ContentType = "text/plain; version=0.0.4; charset=utf-8";
                var text = Prometheus.Render([.. ctx.LoadedChatModels().Select(c => (c.Name, c.Instance.Engine.GetMetrics()))]) + Prometheus.RenderHost(ctx.Host);
                await http.Response.WriteAsync(text);
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
            if (ctx.Host.Get(id) is not { } info)
            {
                await ApiErrors.WriteAsync(http, 404, anthropic ? "not_found_error" : "invalid_request_error", $"The model '{id}' does not exist.", code: "model_not_found");
                return;
            }
            await HttpJson.WriteAsync(http, anthropic ? AnthropicApi.ModelObject(info) : OpenAiApi.ModelObject(info));
        });
    }

    /// <summary>Loaded chat models and their instances.</summary>
    internal static List<(string Name, ChatModelInstance Instance)> LoadedChatModels(this SharplightServerContext ctx)
        => [.. ctx.Host.LoadedInstances().Where(x => x.Instance is ChatModelInstance).Select(x => (x.Name, (ChatModelInstance)x.Instance))];
}
