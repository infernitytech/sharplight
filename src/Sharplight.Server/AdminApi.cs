using System.Net;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.DependencyInjection;
using Sharplight.Llm.Serving.Hosting;

namespace Sharplight.Server;

/// <summary>
/// Model management: <c>GET /admin/models</c> (state, memory per model and device), <c>POST /admin/models/{name}/load</c> and
/// <c>POST /admin/models/{name}/unload[?force=true]</c>. Protected by the API keys when any are configured; without keys only
/// loopback callers are let in (the API-key middleware has nothing to check).
/// </summary>
internal static class AdminApi
{
    extension(IEndpointRouteBuilder routes)
    {
        /// <summary>Maps the <c>/admin/models</c> endpoints (nothing when <see cref="SharplightServerOptions.Admin"/> is off).</summary>
    public IEndpointRouteBuilder MapAdmin()
        {
            var ctx = routes.ServiceProvider.GetRequiredService<SharplightServerContext>();
            if (!ctx.Options.Admin)
            {
                return routes;
            }

            var host = ctx.Host;

            routes.MapGet("/admin/models", async http =>
            {
                if (!await Allowed(http, ctx))
                {
                    return;
                }

                await HttpJson.WriteAsync(http, Describe(host));
            });
            routes.MapPost("/admin/models/{name}/load", async http =>
            {
                if (!await Allowed(http, ctx))
                {
                    return;
                }

                string name = (string)http.Request.RouteValues["name"]!;
                await host.LoadAsync(name, http.RequestAborted);   // errors map to 404 / 503 through the shared handler
                await HttpJson.WriteAsync(http, ModelJson(host.Get(name)!));
            });
            routes.MapPost("/admin/models/{name}/unload", async http =>
            {
                if (!await Allowed(http, ctx))
                {
                    return;
                }

                string name = (string)http.Request.RouteValues["name"]!;
                bool force = http.Request.Query.TryGetValue("force", out var f) && f.ToString() is "1" or "true";
                await host.UnloadAsync(name, null, force);
                await HttpJson.WriteAsync(http, ModelJson(host.Get(name)!));
            });
            return routes;
        }
    }

    private static async Task<bool> Allowed(HttpContext http, SharplightServerContext ctx)
    {
        if (ctx.Options.ApiKeys.Count > 0)
        {
            return true;   // the middleware already checked the key
        }

        var ip = http.Connection.RemoteIpAddress;
        if (ip is null || IPAddress.IsLoopback(ip))
        {
            return true;
        }

        await ApiErrors.WriteAsync(http, 403, "permission_error", "Admin endpoints need an API key (or a local caller).", "forbidden");
        return false;
    }

    internal static JsonObject ModelJson(ModelInfo m) => new()
    {
        ["id"] = m.Name, ["kind"] = m.Kind.ToString().ToLowerInvariant(), ["state"] = m.State.ToString().ToLowerInvariant(),
        ["pinned"] = m.Pinned, ["load_at_startup"] = m.LoadAtStartup, ["resident_bytes"] = m.ResidentBytes, ["estimated_bytes"] = m.EstimatedBytes,
        ["active_requests"] = m.ActiveLeases, ["loaded_at"] = m.LoadedAt, ["last_used"] = m.LastUsed, ["error"] = m.Error, ["description"] = m.Description,
    };

    internal static JsonObject Describe(ModelHost host)
    {
        var devices = new JsonArray();
        foreach (var (d, i) in host.MemoryPools())
        {
            devices.Add(new JsonObject
            {
                ["device"] = d.Name, ["total_bytes"] = i.Total, ["budget_bytes"] = i.Budget, ["used_bytes"] = i.Usage,
                ["used_by_sharplight_bytes"] = i.UsedByUs, ["available_bytes"] = i.Available, ["exact"] = i.IsExact,
            });
        }

        return new JsonObject
        {
            ["models"] = new JsonArray(host.List().Select(m => (JsonNode)ModelJson(m)).ToArray()),
            ["devices"] = devices,
            ["reserve_bytes"] = host.Options.ReserveBytes, ["max_loaded_models"] = host.Options.MaxLoadedModels,
        };
    }
}
