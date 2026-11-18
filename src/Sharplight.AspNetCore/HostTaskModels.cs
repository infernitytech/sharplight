using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Sharplight.Hosting.Loaders;
using Sharplight.Llm.Serving.Hosting;
using Sharplight.Tasks;

namespace Sharplight.AspNetCore;

/// <summary>Looks up the task clients of the server's <see cref="ModelHost"/> for the media endpoints (<see cref="SharplightServerOptions.ServeHostModels"/>).</summary>
internal static class HostTaskModels
{
    /// <summary>True when the endpoints serve the host's models (decided per request, so models loaded later count).</summary>
    public static bool Enabled(IServiceProvider services) => services.GetService<SharplightServerContext>()?.Options.ServeHostModels ?? false;

    /// <summary>
    /// Leases the host's task model <paramref name="requested"/> and returns its <typeparamref name="TClient"/>; <see langword="null"/> when the
    /// host does not serve such a name (the caller falls back to registered models). Dispose the lease after the response is written.
    /// </summary>
    /// <exception cref="ApiException">The model exists but is of another kind.</exception>
    public static async Task<(TClient Client, IDisposable Lease)?> TryAcquireAsync<TClient>(HttpContext http, string? requested, string what)
        where TClient : class, ITaskClient
    {
        var ctx = http.RequestServices.GetService<SharplightServerContext>();
        if (ctx is null || !ctx.Options.ServeHostModels || string.IsNullOrEmpty(requested) || !ctx.Host.Contains(requested) || !ctx.IsServed(requested))
        {
            return null;
        }

        if (ctx.Host.Get(requested)?.Kind != ModelKind.Task)
        {
            return null;
        }

        var lease = await ctx.AcquireAsync(http, requested, ModelKind.Task);
        if (lease.As<LoadedModel>() is ITaskClientModel task && task.FindClient<TClient>() is { } client)
        {
            return (client, lease);
        }

        lease.Dispose();
        throw ApiException.ModelNotFound($"The model '{requested}' is not a {what} model.", ApiErrors.IsAnthropic(http));
    }
}
