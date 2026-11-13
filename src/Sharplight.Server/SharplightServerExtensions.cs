using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.DependencyInjection;
using Sharplight.Llm.Serving;
using Sharplight.Llm.Serving.Chat;
using Sharplight.Llm.Serving.Hosting;

namespace Sharplight.Server;

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

    private static bool KeyIsValid(HttpContext http, List<string> keys)
    {
        string? presented = null;
        string auth = http.Request.Headers.Authorization.ToString();
        if (auth.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase))
        {
            presented = auth[7..].Trim();
        }
        else if (http.Request.Headers.TryGetValue("x-api-key", out var k))
        {
            presented = k.ToString();
        }

        if (presented is null)
        {
            return false;
        }

        var pb = System.Text.Encoding.UTF8.GetBytes(presented);
        bool ok = false;
        foreach (var key in keys)
        {
            ok |= System.Security.Cryptography.CryptographicOperations.FixedTimeEquals(pb, System.Text.Encoding.UTF8.GetBytes(key));
        }

        return ok;
    }

    /// <summary>
    /// Maps everything the options enable: <c>/health</c>, <c>/metrics</c>, the OpenAI and the Anthropic API, plus the retrieval
    /// endpoints of <see cref="RetrievalApi"/> when embedding / rerank models are registered.
    /// </summary>
    public static IEndpointRouteBuilder MapSharplight(this IEndpointRouteBuilder routes)
    {
        var o = routes.ServiceProvider.GetRequiredService<SharplightServerContext>().Options;
        routes.MapOperations();
        routes.MapAdmin();
        if (o.OpenAi)
        {
            routes.MapOpenAi();
        }

        if (o.Anthropic)
        {
            routes.MapAnthropic();
        }

        routes.MapSharplightRetrieval();   // /v1/embeddings, /v1/rerank when such models are registered
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
        {
            routes.MapGet("/metrics", async http =>
            {
                http.Response.ContentType = "text/plain; version=0.0.4; charset=utf-8";
                var text = Prometheus.Render([.. ctx.LoadedChatModels().Select(c => (c.Name, c.Instance.Engine.GetMetrics()))]) + Prometheus.RenderHost(ctx.Host);
                await http.Response.WriteAsync(text);
            });
        }

        return routes;
    }

    internal static void MapModels(this IEndpointRouteBuilder routes, SharplightServerContext ctx)
    {
        if (ctx.ModelsMapped)
        {
            return;
        }

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
    internal static List<(string Name, IChatModel Instance)> LoadedChatModels(this SharplightServerContext ctx)
        => [.. ctx.Host.LoadedInstances().Where(x => x.Instance is IChatModel).Select(x => (x.Name, (IChatModel)x.Instance))];
}
