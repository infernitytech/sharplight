using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Sharplight.Audio;
using Sharplight.Hub;
using Sharplight.Storage;
using Sharplight.Studio.Services.Audio;
using Sharplight.Studio.Services.Documents;
using Sharplight.Studio.Services.Feed;
using Sharplight.Studio.Services.Fit;
using Sharplight.Studio.Services.Hub;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.Services.Maintenance;
using Sharplight.Studio.Services.Monitoring;
using Sharplight.Studio.Services.Pages;
using Sharplight.Studio.Services.Platform;
using Sharplight.Studio.Services.Precompile;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.Services.Server;
using Sharplight.Studio.Services.Settings;
using Sharplight.Studio.Services.Storage;
using Sharplight.Studio.Services.Updates;

namespace Sharplight.Studio.Services;

/// <summary>The composition root: creates the real services and disposes them when the app closes.</summary>
internal sealed class StudioHost : IDisposable
{
    private readonly EngineModelRuntime _runtime;
    private readonly HubService _hub;
    private readonly StatsRecorder _recorder;
    private readonly ModelUsageRecorder _usageRecorder;
    private readonly KestrelApiServer _server;
    private readonly ServiceProvider _container;

    private StudioHost(StudioServices services, EngineModelRuntime runtime, HubService hub, StatsRecorder recorder, ModelUsageRecorder usageRecorder, KestrelApiServer server,
        ServiceProvider container)
    {
        Services = services;
        _container = container;
        _runtime = runtime;
        _hub = hub;
        _recorder = recorder;
        _usageRecorder = usageRecorder;
        _server = server;
    }

    /// <summary>The services the view models use.</summary>
    public StudioServices Services { get; }

    /// <summary>Creates every service; <paramref name="modelsDirectory"/> is the <c>--models-dir</c> argument.</summary>
    public static StudioHost Create(string? modelsDirectory, IPlatformServices platform, IUiDispatcher dispatcher, IThemeService theme)
    {
        // The storage layout is known before the settings load: it says where they are. A move into this layout finishes here, before
        // anything opens the files it copied.
        var environment = StorageEnvironment.Process;
        var layout = StorageLayout.Resolve(environment);
        StorageMigration.CompletePending(layout);
        var paths = StudioPaths.From(layout, environment.GetVariable(StudioPaths.HomeVariable));
        var settings = new JsonSettingsService(paths.SettingsFile, modelsDirectory, layout);
        var storage = new StorageLocationService(environment, layout, settings);
        var container = new ServiceCollection().AddSingleton<ISettingsService>(settings).AddSingleton<IStorageLocationService>(storage)
            .AddSingleton(Program.Logs).AddSingleton(typeof(ILogger<>), typeof(Logger<>)).AddSingleton(paths).AddSharplightStorage(layout).AddStudioUpdates().AddStudioFeed().AddSharplight().AddStudioProviders().BuildServiceProvider();
        var engine = EngineServices.From(container);
        var audioDecoder = container.GetRequiredService<IAudioDecoder>();
        var audioEncoder = container.GetRequiredService<IAudioEncoder>();
        var audioDevices = container.GetRequiredService<IAudioDeviceProvider>();
        var precompiler = new PrecompileService(settings, engine);
        var runtime = new EngineModelRuntime(engine, () => settings.Current.Residency, name => ResolveOnDemand(settings, name, precompiler.IsBusy),
            () => settings.Current.UseCompilationCache,
            Program.Logs.CreateLogger<EngineModelRuntime>(), () => settings.Current.DeviceProviders);
        var hub = new HubService(settings, engine, paths.HighlightsFile, paths.CatalogFile);
        var database = new StudioDatabase(paths.DatabaseFile);
        var stats = new SqliteStatsStore(database);
        var server = new KestrelApiServer(runtime.Host);
        var library = new ModelLibrary(settings, engine.Converter);
        var usage = new JsonModelUsageStore(paths.ModelUsageFile);
        var services = new StudioServices(settings, runtime, hub, library, new SqliteConversationStore(database), stats, server,
            new DocumentExtractor(), new SystemMonitor(runtime), platform, dispatcher, theme,
            container.GetRequiredService<IUpdateService>(), precompiler, new FilePageStorage(paths.PagesDirectory, paths.GenerationsDirectory),
            new EngineAudioPlayer(audioDecoder, audioDevices, settings), new EngineAudioRecorder(audioDevices, audioEncoder, settings),
            audioDecoder, audioDevices, storage, container.GetRequiredService<IFeedService>(), usage,
            new ModelMaintenanceService(settings, storage, runtime, library, usage, precompiler), new LicenseGate(settings),
            new DeviceMemoryWatch(runtime, settings));
        return new StudioHost(services, runtime, hub, new StatsRecorder(runtime, stats), new ModelUsageRecorder(runtime, usage), server, container);
    }

    /// <summary>
    /// A library model a chat or API client asked for by name while it is not resident: stored models by <c>owner/name[:tag]</c> or
    /// alias, added files by file name. Null when on-demand loading is off or nothing matches.
    /// </summary>
    internal static ModelLoadRequest? ResolveOnDemand(ISettingsService settings, string name, Func<string, bool>? isPrecompiling = null)
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
        // A package being precompiled is rewritten in place: it is not loaded until that finished.
        if (path is null || isPrecompiling?.Invoke(path) == true)
        {
            return null;
        }

        return new ModelLoadRequest(name, path, current.DefaultLoad, current.DisabledFeatures);
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
        _usageRecorder.Dispose();
        _runtime.Dispose();
        _hub.Dispose();
        _container.Dispose();
    }
}
