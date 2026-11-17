using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Options;

namespace Sharplight.Hosting;

/// <summary>Loads the startup (non-lazy) models when the application starts (<see cref="ModelHostingOptions.LoadStartupModels"/>).</summary>
internal sealed class HostedModelStartup(IModelHostManager manager, IOptions<ModelHostingOptions> options) : IHostedService
{
    public async Task StartAsync(CancellationToken cancellationToken)
    {
        if (options.Value.LoadStartupModels)
        {
            await manager.Host.LoadStartupModelsAsync(cancellationToken);
        }
    }

    public Task StopAsync(CancellationToken cancellationToken) => Task.CompletedTask;
}
