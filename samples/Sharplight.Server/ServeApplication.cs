using Sharplight.Runtime;
using Sharplight.Samples;

namespace Sharplight.Server;

/// <summary>Plans the served models, pulls the ones missing from the store, loads them and runs the HTTP server until cancelled.</summary>
internal static class ServeApplication
{
    public static async Task<int> RunAsync(ServeSettings settings, CancellationToken cancellationToken)
    {
        var plan = ServerPlanBuilder.Build(settings);
        await ResolveModelsAsync(plan, settings, cancellationToken);

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

    /// <summary>Replaces stored and Hub model names by package paths, pulling models that are not stored yet.</summary>
    private static async Task ResolveModelsAsync(ServerPlan plan, ServeSettings settings, CancellationToken cancellationToken)
    {
        if (plan.Models.All(model => ModelResolver.IsLocalPath(model.Path)))
        {
            return;
        }

        using var hub = new HubContext(settings.StoreRoot);
        var resolver = new ModelResolver(hub, settings.Pull);
        foreach (var model in plan.Models)
        {
            model.Path = await resolver.ResolveAsync(model.Path, cancellationToken);
        }
    }
}
