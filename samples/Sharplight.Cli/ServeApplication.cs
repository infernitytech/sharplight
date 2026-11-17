using Sharplight.Hosting;
using Sharplight.Samples;
using Sharplight.Samples.Hub;

namespace Sharplight.Cli;

/// <summary><c>serve</c>: builds the hosting options, pulls the models missing from the store and runs the HTTP server until cancelled.</summary>
internal static class ServeApplication
{
    public static async Task<int> RunAsync(SampleRuntime runtime, ServeSettings settings, CancellationToken cancellationToken)
    {
        var options = HostingOptionsBuilder.Build(settings);
        await ResolveModelsAsync(runtime, options, settings, cancellationToken);
        await HttpServer.RunAsync(runtime, options, settings, cancellationToken);
        return ExitCodes.Success;
    }

    /// <summary>Replaces stored and Hub model names by package paths, pulling models that are not stored yet.</summary>
    private static async Task ResolveModelsAsync(SampleRuntime runtime, ModelHostingOptions options, ServeSettings settings, CancellationToken cancellationToken)
    {
        if (options.Models.All(model => ModelResolver.IsLocalPath(model.Path)))
        {
            return;
        }

        using var hub = new HubContext(runtime, settings.StoreRoot);
        var resolver = new ModelResolver(hub, settings.Pull);
        foreach (var model in options.Models)
        {
            model.Path = await resolver.ResolveAsync(model.Path, cancellationToken);
        }
    }
}
