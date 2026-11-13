using Microsoft.Extensions.Hosting;

namespace Sharplight.Server;

/// <summary>Drains the models when the host stops so running requests can finish while new ones get 503.</summary>
internal sealed class EngineLifetime(SharplightServerContext ctx) : IHostedService
{
    public Task StartAsync(CancellationToken ct) => Task.CompletedTask;
    public async Task StopAsync(CancellationToken ct) => await ctx.Host.DrainAsync(ctx.Options.DrainTimeout);
}
