using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.DependencyInjection;
using Sharplight.Llm.Serving;

namespace Sharplight.AspNetCore;

/// <summary>
/// Wraps every API endpoint of a <c>MapSharplightServer</c> call: authenticates (<see cref="IApiKeyValidator"/>), resolves the request
/// (<see cref="IModelEndpointResolver"/>), runs the handler with the result available to it, maps errors to API error bodies, releases
/// admission handles and reports usage and metrics.
/// </summary>
internal sealed class SharplightRequestPipeline(IServiceProvider services, SharplightEndpointOptions endpoint)
{
    private readonly SharplightServerOptions? _server = services.GetService<SharplightServerContext>()?.Options;
    private readonly SharplightServerMetrics? _metrics = services.GetService<SharplightServerMetrics>();
    private readonly IReadOnlyList<ISharplightUsageObserver> _observers =
        [.. services.GetServices<ISharplightUsageObserver>(), .. endpoint.UsageObserver is { } observer ? new[] { observer } : []];

    private IApiKeyValidator Validator => endpoint.KeyValidator ?? services.GetService<IApiKeyValidator>() ?? new StaticApiKeyValidator(_server?.ApiKeys ?? []);

    private IModelEndpointResolver Resolver => endpoint.Resolver ?? services.GetService<IModelEndpointResolver>() ?? new DefaultModelEndpointResolver();

    public RequestDelegate Wrap(RequestDelegate inner) => http => RunAsync(http, inner);

    private async Task RunAsync(HttpContext http, RequestDelegate inner)
    {
        if (HttpMethods.IsOptions(http.Request.Method))
        {
            await inner(http);
            return;
        }

        SharplightRequestFeature? feature = null;
        var started = System.Diagnostics.Stopwatch.GetTimestamp();
        try
        {
            var authentication = await Validator.ValidateAsync(Tenants.PresentedKey(http), http, http.RequestAborted);
            if (!authentication.IsValid)
            {
                await ApiErrors.WriteAsync(http, 401, "authentication_error", "Invalid or missing API key.", code: "invalid_api_key");
                return;
            }

            var context = await Resolver.ResolveAsync(http, authentication, http.RequestAborted);
            feature = new SharplightRequestFeature(context, endpoint) { Admission = endpoint.Admission ?? services.GetService<IRequestAdmission>() };
            http.Features.Set(feature);
            await inner(http);
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
        finally
        {
            if (feature is not null)
            {
                await feature.ReleaseAsync();
            }

            Report(http, feature, System.Diagnostics.Stopwatch.GetElapsedTime(started));
        }
    }

    private void Report(HttpContext http, SharplightRequestFeature? feature, TimeSpan elapsed)
    {
        var usage = new SharplightUsage
        {
            Api = (http.GetEndpoint() as RouteEndpoint)?.RoutePattern.RawText ?? http.Request.Path.Value ?? "",
            Model = feature?.Model,
            Scope = feature?.Context.Scope,
            Principal = feature?.Context.Principal,
            Context = feature?.Context,
            InputTokens = feature?.InputTokens ?? 0,
            CachedInputTokens = feature?.CachedInputTokens ?? 0,
            OutputTokens = feature?.OutputTokens ?? 0,
            Streaming = feature?.Streaming ?? false,
            StatusCode = http.Response.StatusCode,
            Duration = elapsed,
            TimeToFirstToken = feature?.TimeToFirstToken,
        };
        _metrics?.Record(usage);
        foreach (var observer in _observers)
        {
            try
            {
                observer.OnRequestCompleted(usage);
            }
            catch (Exception)
            {
                // An observer must not break the response; it reports its own failures.
            }
        }
    }
}
