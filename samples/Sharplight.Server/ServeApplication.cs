using Sharplight.Runtime;
using Sharplight.Samples;

namespace Sharplight.Server;

/// <summary>Plans the served models, loads them and runs the HTTP server until cancelled.</summary>
internal static class ServeApplication
{
    public static async Task<int> RunAsync(ServeSettings settings, CancellationToken cancellationToken)
    {
        var plan = ServerPlanBuilder.Build(settings);

        // The pool outlives the host: engines must stop before the devices they run on are disposed.
        using var devices = new DevicePool();
        using var host = new ModelHostBuilder(settings, plan, devices).Build();

        try
        {
            await host.LoadStartupModelsAsync(cancellationToken);
        }
        catch (Exception error) when (error is not OperationCanceledException)
        {
            throw new SampleException($"Cannot load model: {error.Message}", error);
        }

        await HttpServer.RunAsync(host, settings, plan, cancellationToken);
        return ExitCodes.Success;
    }
}
