using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.DependencyInjection;
using Sharplight.Llm.Serving;
using Sharplight.Llm.Serving.Hosting;

namespace Sharplight.AspNetCore;

/// <summary>
/// ASP.NET Core integration: adds the cross-cutting middleware and maps the
/// OpenAI-compatible (<c>/v1/chat/completions</c>, <c>/v1/completions</c>, <c>/v1/models</c>), Anthropic-compatible
/// (<c>/v1/messages</c>), embedding / rerank (see <see cref="RetrievalApi"/>), admin, health and metrics endpoints. The models and
/// options are registered with <c>services.AddSharplightServer(...)</c>.
/// </summary>
/// <example>
/// A single-model chat server:
/// <code>
/// var builder = WebApplication.CreateBuilder(args);
/// using var engine = new InferenceEngine(model.CreateBatchDecoder(slots: 4), model.Tokenizer);
/// var service = new ModelService(engine, model.ChatTemplate, new ModelServiceOptions { ModelName = "qwen3" });
/// builder.Services.AddSharplightServer(service, o => o.ApiKeys.Add("secret"));
///
/// var app = builder.Build();
/// app.UseSharplightServer();   // CORS, API keys, error mapping
/// app.MapSharplightServer();   // health, metrics, OpenAI and Anthropic APIs
/// app.Run();
/// </code>
/// </example>
public static class SharplightServerExtensions
{
    extension(IApplicationBuilder app)
    {
        /// <summary>
        /// Adds CORS handling, API-key authentication and the mapping of API errors (<see cref="ApiException"/>,
        /// <see cref="EngineException"/>) to JSON error bodies in front of the endpoints. Call it before
        /// <see cref="MapSharplightServer"/>; <c>/health</c> and <c>/metrics</c> stay open when API keys are configured.
        /// </summary>
        /// <exception cref="InvalidOperationException"><c>AddSharplightServer</c> was not called.</exception>
        public IApplicationBuilder UseSharplightServer()
        {
            var ctx = app.ApplicationServices.GetRequiredService<SharplightServerContext>();
            app.Use((http, next) => HandleRequestAsync(ctx, http, next));
            return app;
        }
    }

    extension(IEndpointRouteBuilder routes)
    {
        /// <summary>
        /// Maps everything the options enable: <c>/health</c>, <c>/metrics</c>, the admin endpoints, the OpenAI and the Anthropic
        /// API, plus the retrieval endpoints of <see cref="RetrievalApi"/> when embedding / rerank models are registered and the audio
        /// endpoints of <see cref="AudioApi"/> when speech recognizers are registered. The API endpoints run through the Sharplight
        /// request pipeline: authentication (<see cref="IApiKeyValidator"/>), per-request resolution (<see cref="IModelEndpointResolver"/>:
        /// models, defaults, tools, isolation scope), admission (<see cref="IRequestAdmission"/>), usage reporting
        /// (<see cref="ISharplightUsageObserver"/>, <see cref="SharplightServerMetrics"/>) and error mapping. Call it on a route group to
        /// serve the APIs under a prefix, several times with different <paramref name="configure"/> hooks for several endpoints or tenants.
        /// </summary>
        /// <param name="configure">Adjusts the <see cref="SharplightEndpointOptions"/> of this mapping, or <see langword="null"/>.</param>
        /// <returns>The route builder.</returns>
        /// <exception cref="InvalidOperationException"><c>AddSharplightServer</c> was not called.</exception>
        /// <example>
        /// <code>
        /// app.MapSharplightServer();                                        // the single-host default at the root
        /// app.MapGroup("/e/{endpoint}").MapSharplightServer(o =>            // per endpoint: own keys, models and defaults
        /// {
        ///     o.Operations = false;
        ///     o.Admin = false;
        ///     o.KeyValidator = endpointKeys;
        ///     o.Resolver = endpointResolver;
        /// });
        /// </code>
        /// </example>
        public IEndpointRouteBuilder MapSharplightServer(Action<SharplightEndpointOptions>? configure = null)
        {
            var ctx = routes.ServiceProvider.GetRequiredService<SharplightServerContext>();
            var o = ctx.Options;
            var endpoint = new SharplightEndpointOptions();
            configure?.Invoke(endpoint);
            if (endpoint.Operations)
            {
                routes.MapOperations();
            }

            if (endpoint.Admin)
            {
                routes.MapAdmin();
            }

            // Every API endpoint below goes through the request pipeline (wrapped after the handler is built).
            var pipeline = new SharplightRequestPipeline(routes.ServiceProvider, endpoint);
            var api = routes.MapGroup("");
            api.WithMetadata(SharplightApiEndpointMetadata.Instance);
            ((IEndpointConventionBuilder)api).Finally(builder =>
            {
                if (builder.RequestDelegate is { } inner)
                {
                    builder.RequestDelegate = pipeline.Wrap(inner);
                }
            });
            bool openAi = endpoint.OpenAi ?? o.OpenAi;
            bool anthropic = endpoint.Anthropic ?? o.Anthropic;
            if (openAi || anthropic)
            {
                api.MapModels(ctx, anthropic);
            }

            if (openAi)
            {
                api.MapOpenAi();
            }

            if (anthropic)
            {
                api.MapAnthropic();
            }

            if (endpoint.Retrieval)
            {
                api.MapSharplightRetrieval();
            }

            if (endpoint.Media)
            {
                api.MapSharplightAudio();
                api.MapSharplightImages();
            }

            return routes;
        }

        /// <summary>
        /// <c>GET /health</c> (status summary), <c>GET /health/live</c> (liveness: the process serves HTTP),
        /// <c>GET /health/ready</c> (readiness: not draining and a model is loaded or loadable; 503 otherwise) and
        /// (if enabled) <c>GET /metrics</c>.
        /// </summary>
        internal IEndpointRouteBuilder MapOperations()
        {
            var ctx = routes.ServiceProvider.GetRequiredService<SharplightServerContext>();
            routes.MapGet("/health", http => WriteHealthAsync(ctx, http));
            routes.MapGet("/health/live", http =>
            {
                http.Response.ContentType = "application/json";
                return http.Response.WriteAsync("{\"status\":\"alive\"}");
            });
            routes.MapGet("/health/ready", http => WriteReadinessAsync(ctx, http));
            if (ctx.Options.Metrics)
            {
                routes.MapGet("/metrics", async http =>
                {
                    http.Response.ContentType = "text/plain; version=0.0.4; charset=utf-8";
                    var perModel = ctx.LoadedChatModels().Select(c => (c.Name, c.Instance.Engine.GetMetrics()));
                    await http.Response.WriteAsync(Prometheus.Render([.. perModel]) + Prometheus.RenderHost(ctx.Host));
                });
            }

            return routes;
        }

        /// <summary><c>GET /v1/models</c> and <c>/v1/models/{id}</c>, shared by the OpenAI and Anthropic APIs, listing what the request may see.</summary>
        internal void MapModels(SharplightServerContext ctx, bool anthropicApi)
        {
            routes.MapGet("/v1/models", async http =>
            {
                bool anthropic = ApiErrors.IsAnthropic(http) && anthropicApi;
                await HttpJson.WriteAsync(http, anthropic ? AnthropicApi.ModelList(ctx, http) : OpenAiApi.ModelList(ctx, http));
            });
            routes.MapGet("/v1/models/{id}", async http =>
            {
                bool anthropic = ApiErrors.IsAnthropic(http) && anthropicApi;
                string id = (string)http.Request.RouteValues["id"]!;
                if (ctx.FindModel(http, id) is not { } info)
                {
                    await ApiErrors.WriteAsync(http, 404, anthropic ? "not_found_error" : "invalid_request_error", $"The model '{id}' does not exist.", code: "model_not_found");
                    return;
                }

                await HttpJson.WriteAsync(http, anthropic ? AnthropicApi.ModelObject(info) : OpenAiApi.ModelObject(info));
            });
        }
    }

    extension(SharplightServerContext ctx)
    {
        /// <summary>Loaded chat models and their instances.</summary>
        internal List<(string Name, IChatModel Instance)> LoadedChatModels()
            => [.. ctx.Host.LoadedInstances().Where(x => x.Instance is IChatModel).Select(x => (x.Name, (IChatModel)x.Instance))];
    }

    private static async Task HandleRequestAsync(SharplightServerContext ctx, HttpContext http, RequestDelegate next)
    {
        var options = ctx.Options;
        if (ApplyCors(options, http))
        {
            return;
        }

        var path = http.Request.Path.Value ?? "";
        bool open = path is "/health" or "/health/live" or "/health/ready" or "/metrics" || HttpMethods.IsOptions(http.Request.Method);
        // The API endpoints authenticate in their own pipeline (with the hooks of their MapSharplightServer call).
        bool pipeline = http.GetEndpoint()?.Metadata.GetMetadata<SharplightApiEndpointMetadata>() is not null;
        if (options.ApiKeys.Count > 0 && !open && !pipeline && !KeyIsValid(http, options.ApiKeys))
        {
            await ApiErrors.WriteAsync(http, 401, "authentication_error", "Invalid or missing API key.", code: "invalid_api_key");
            return;
        }

        try
        {
            await next(http);
        }
        catch (ApiException ex)
        {
            await ApiErrors.WriteAsync(http, ex);
        }
        catch (EngineException ex)
        {
            await ApiErrors.WriteAsync(http, ex);
        }
        catch (OperationCanceledException) when (http.RequestAborted.IsCancellationRequested)
        {
            // The client went away; there is nobody to answer.
        }
        catch (Exception ex) when (!http.Response.HasStarted)
        {
            await ApiErrors.WriteAsync(http, 500, "server_error", "Internal error: " + ex.Message);
        }
    }

    /// <summary>Adds the CORS response headers; returns true when the request was a preflight that is now fully answered.</summary>
    private static bool ApplyCors(SharplightServerOptions options, HttpContext http)
    {
        if (options.CorsOrigins.Count == 0 || http.Request.Headers.Origin.Count == 0)
        {
            return false;
        }

        string origin = http.Request.Headers.Origin!;
        bool anyOrigin = options.CorsOrigins.Contains("*");
        if (anyOrigin || options.CorsOrigins.Contains(origin, StringComparer.OrdinalIgnoreCase))
        {
            http.Response.Headers.AccessControlAllowOrigin = anyOrigin ? "*" : origin;
            http.Response.Headers.Vary = "Origin";
            http.Response.Headers.AccessControlExposeHeaders = "*";
        }

        if (!HttpMethods.IsOptions(http.Request.Method))
        {
            return false;
        }

        http.Response.Headers.AccessControlAllowMethods = "GET, POST, OPTIONS";
        http.Response.Headers.AccessControlAllowHeaders = http.Request.Headers.AccessControlRequestHeaders.Count > 0
            ? http.Request.Headers.AccessControlRequestHeaders
            : "*";
        http.Response.Headers.AccessControlMaxAge = "600";
        http.Response.StatusCode = StatusCodes.Status204NoContent;
        return true;
    }

    private static bool KeyIsValid(HttpContext http, IList<string> keys)
    {
        string? presented = Tenants.PresentedKey(http);

        if (presented is null)
        {
            return false;
        }

        var presentedBytes = Encoding.UTF8.GetBytes(presented);
        bool ok = false;
        foreach (var key in keys)
        {
            // Every key is compared (no early exit) so timing does not reveal which one matched.
            ok |= CryptographicOperations.FixedTimeEquals(presentedBytes, Encoding.UTF8.GetBytes(key));
        }

        return ok;
    }

    private static async Task WriteReadinessAsync(SharplightServerContext ctx, HttpContext http)
    {
        var chats = ctx.LoadedChatModels();
        bool draining = chats.Count > 0 && chats.All(c => c.Instance.Engine.IsDraining);
        var models = ctx.ListModels();
        // Lazily loaded models count as ready: the first request loads them. A model still loading at startup does not.
        bool servable = models.Any(m => m.State is Llm.Serving.Hosting.ModelState.Loaded or Llm.Serving.Hosting.ModelState.Unloaded);
        string status = draining ? "draining" : servable ? "ready" : "loading";
        http.Response.StatusCode = status == "ready" ? 200 : 503;
        http.Response.ContentType = "application/json";
        await http.Response.WriteAsync(JsonSerializer.Serialize(new { status }));
    }

    private static async Task WriteHealthAsync(SharplightServerContext ctx, HttpContext http)
    {
        var chats = ctx.LoadedChatModels();
        bool draining = chats.Count > 0 && chats.All(c => c.Instance.Engine.IsDraining);
        var defaultName = ctx.DefaultName();
        var defaultChat = chats.FirstOrDefault(c => c.Name == defaultName);
        var defaultMetrics = defaultChat.Instance?.Engine.GetMetrics();
        http.Response.StatusCode = draining ? 503 : 200;
        http.Response.ContentType = "application/json";
        await http.Response.WriteAsync(JsonSerializer.Serialize(new
        {
            status = draining ? "draining" : "ok",
            model = defaultName,
            slots = defaultMetrics?.SlotCount ?? 0,
            active = chats.Sum(c => c.Instance.Engine.GetMetrics().ActiveSequences),
            queued = chats.Sum(c => c.Instance.Engine.GetMetrics().QueueDepth),
            models = ctx.ListModels().Select(i => new { id = i.Name, kind = i.Kind.ToString().ToLowerInvariant(), state = i.State.ToString().ToLowerInvariant() }),
        }));
    }
}
