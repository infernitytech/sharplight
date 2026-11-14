using Microsoft.Extensions.Hosting;

namespace Sharplight.AspNetCore;

/// <summary>Drains the models when the host stops so running requests can finish while new ones get 503.</summary>
internal sealed class EngineLifetime(SharplightServerContext ctx) : IHostedService
{
    public Task StartAsync(CancellationToken ct) => Task.CompletedTask;
    public async Task StopAsync(CancellationToken ct)
    {
        if (ctx.Options.DrainModelsOnStop)
        {
            await ctx.Host.DrainAsync(ctx.Options.DrainTimeout);
        }
    }
}
