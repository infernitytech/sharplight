using Sharplight.Hub;
using Sharplight.Studio.Services.Documents;
using Sharplight.Studio.Services.Hub;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.Services.Monitoring;
using Sharplight.Studio.Services.Platform;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.Services.Server;
using Sharplight.Studio.Services.Settings;
using Sharplight.Studio.Services.Storage;

namespace Sharplight.Studio.Services;

/// <summary>The composition root: creates the real services and disposes them when the app closes.</summary>
internal sealed class StudioHost : IDisposable
{
    private readonly EngineModelRuntime _runtime;
    private readonly HubService _hub;
    private readonly StatsRecorder _recorder;
    private readonly KestrelApiServer _server;

    private StudioHost(StudioServices services, EngineModelRuntime runtime, HubService hub, StatsRecorder recorder, KestrelApiServer server)
    {
        Services = services;
        _runtime = runtime;
        _hub = hub;
        _recorder = recorder;
        _server = server;
    }

    /// <summary>The services the view models use.</summary>
    public StudioServices Services { get; }

    /// <summary>Creates every service; <paramref name="modelsDirectory"/> is the <c>--models-dir</c> argument.</summary>
    public static StudioHost Create(string? modelsDirectory, IPlatformServices platform, IUiDispatcher dispatcher, IThemeService theme)
    {
        var paths = StudioPaths.Default();
        var settings = new JsonSettingsService(paths.SettingsFile, modelsDirectory);
        var runtime = new EngineModelRuntime(() => settings.Current.Residency, name => ResolveOnDemand(settings, name));
        var hub = new HubService(settings);
        var database = new StudioDatabase(paths.DatabaseFile);
        var stats = new SqliteStatsStore(database);
        var server = new KestrelApiServer(runtime.Host);
        var services = new StudioServices(settings, runtime, hub, new ModelLibrary(settings), new SqliteConversationStore(database), stats, server,
            new DocumentExtractor(), new SystemMonitor(runtime), platform, dispatcher, theme);
        return new StudioHost(services, runtime, hub, new StatsRecorder(runtime, stats), server);
    }

    /// <summary>
    /// A library model a chat or API client asked for by name while it is not resident: stored models by <c>owner/name[:tag]</c> or
    /// alias, added files by file name. Null when on-demand loading is off or nothing matches.
    /// </summary>
    internal static ModelLoadRequest? ResolveOnDemand(ISettingsService settings, string name)
    {
        var current = settings.Current;
        if (!current.Residency.LoadOnDemand)
        {
            return null;
        }

        string? path = null;
        try
        {
            path = new ModelStore(settings.StoreRoot).Find(name) is { } manifest && File.Exists(manifest.PackagePath) ? manifest.PackagePath : null;
        }
        catch (HubException)
        {
            // Several stored models match: the client must add the tag.
        }

        path ??= current.LocalModels.FirstOrDefault(local => File.Exists(local)
            && string.Equals(System.IO.Path.GetFileNameWithoutExtension(local), name, StringComparison.OrdinalIgnoreCase));
        return path is null ? null : new ModelLoadRequest(name, path, current.DefaultLoad, current.DisabledFeatures);
    }

    /// <summary>Stops the server, unloads the models and releases the devices.</summary>
    public void Dispose()
    {
        try
        {
            _server.StopAsync().Wait(TimeSpan.FromSeconds(5));
        }
        catch (AggregateException)
        {
            // Shutting down regardless.
        }

        _recorder.Dispose();
        _runtime.Dispose();
        _hub.Dispose();
    }
}
